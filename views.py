from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, HttpResponse, FileResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib import messages
from django.utils import timezone
import os
import logging
from django.conf import settings
import json
from .models import Resume
from .enhanced_pdf_parser import PDFParser
from .batch_processor import BatchProcessor
from .config import get_openai_config, get_llm_config, validate_config

logger = logging.getLogger(__name__)

@csrf_exempt
def landing_page(request):
    """Landing page view - main entry point for the website."""
    if request.method == 'POST':
        try:
            resume_files = request.FILES.getlist('resumes')
            job_description = request.POST.get('job_description', '')
            disable_ocr_flag = request.POST.get('disable_ocr', '') in ['1', 'true', 'on', 'yes']
            
            if not resume_files:
                return JsonResponse({'error': 'No resume files uploaded'}, status=400)
            
            if not job_description:
                return JsonResponse({'error': 'Job description is required'}, status=400)
            
            if len(resume_files) > 25:
                return JsonResponse({'error': 'Maximum 25 resumes allowed'}, status=400)
            
            # Get API key from provider-agnostic configuration
            llm_cfg = get_llm_config()
            api_key = llm_cfg['api_key']
            
            # Check configuration status
            config_issues = validate_config()
            if config_issues:
                logger.warning("Configuration issues detected: %s", config_issues)
            
            temp_paths = []
            for resume_file in resume_files:
                file_path = os.path.join(settings.MEDIA_ROOT, 'temp_uploads', resume_file.name)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, 'wb+') as destination:
                    for chunk in resume_file.chunks():
                        destination.write(chunk)
                temp_paths.append(file_path)
            
            # Initialize batch processor with API key
            processor = BatchProcessor(api_key=api_key, disable_ocr=disable_ocr_flag)
            
            # Process the batch
            results = processor.process_batch(temp_paths, job_description)
            
            # Store batch results in session for ranking view
            if 'batch_results' not in request.session:
                request.session['batch_results'] = []
            
            # Add current batch results with timestamp
            batch_info = {
                'timestamp': timezone.now().isoformat(),
                'job_description': job_description,
                'resume_count': len(resume_files),
                'results': results,
                'filenames': [f.name for f in resume_files],
                'disable_ocr': disable_ocr_flag
            }
            request.session['batch_results'].append(batch_info)
            # Ensure Django persists in-place session mutations
            request.session.modified = True
            
            # Keep only last 5 batches to avoid session bloat
            if len(request.session['batch_results']) > 5:
                request.session['batch_results'] = request.session['batch_results'][-5:]
                request.session.modified = True
            
            # Clean up temporary files
            for path in temp_paths:
                try:
                    os.remove(path)
                except:
                    pass
            
            return JsonResponse(results)
            
        except Exception as e:
            logger.error(f"Error in batch upload: {str(e)}")
            return JsonResponse({'error': str(e)}, status=500)
    
    # GET request - show the landing page
    llm_cfg = get_llm_config()
    config_issues = validate_config()
    
    context = {
        'api_key_configured': bool(llm_cfg['api_key']),
        'config_issues': config_issues,
        'llm_provider': llm_cfg.get('provider', 'openai'),
        'llm_model': llm_cfg.get('model', ''),
        'llm_enabled': llm_cfg.get('enabled', False)
    }
    
    return render(request, 'resume_processor/landing_page.html', context)

def upload_resume(request):
    """View for uploading resume PDF files."""
    if request.method == 'POST':
        if 'resume_file' in request.FILES:
            resume_file = request.FILES['resume_file']
            
            # Check if file is a PDF
            if not resume_file.name.lower().endswith('.pdf'):
                messages.error(request, 'Please upload a PDF file.')
                return redirect('resume_processor:upload_resume')
            
            # Save the uploaded file
            media_dir = os.path.join(settings.MEDIA_ROOT, 'resumes')
            os.makedirs(media_dir, exist_ok=True)
            
            file_path = os.path.join(media_dir, resume_file.name)
            with open(file_path, 'wb+') as destination:
                for chunk in resume_file.chunks():
                    destination.write(chunk)
            
            # Create Resume instance
            resume = Resume(
                filename=resume_file.name,
                original_file_path=file_path,
                candidate_id=os.path.splitext(resume_file.name)[0],
                processing_status='processing'
            )
            resume.save()
            
            # Process the PDF and extract to JSON
            pdf_parser = PDFParser()
            try:
                json_file_path = pdf_parser.extract_to_json(file_path, resume.filename)
                
                # Load structured data and store in parsed_data field
                import json
                with open(json_file_path, 'r', encoding='utf-8') as f:
                    structured_data = json.load(f)
                
                resume.parsed_data = structured_data
                resume.processing_status = 'completed'
                resume.processed_at = timezone.now()
                resume.save()
                messages.success(request, f'Resume "{resume.filename}" processed successfully!')
            except Exception as e:
                resume.processing_status = 'failed'
                resume.error_message = str(e)
                resume.save()
                messages.error(request, f'Failed to process resume "{resume.filename}": {str(e)}')
            return redirect('resume_processor:resume_list')
    return render(request, 'resume_processor/upload.html', {'show_nav': True})

def resume_list(request):
    """View for listing all processed resumes."""
    resumes = Resume.objects.all()
    return render(request, 'resume_processor/resume_list.html', {'resumes': resumes, 'show_nav': True})

def resume_detail(request, resume_id):
    """View for showing details of a specific resume."""
    try:
        resume = Resume.objects.get(id=resume_id)
        return render(request, 'resume_processor/resume_detail.html', {'resume': resume, 'show_nav': True})
    except Resume.DoesNotExist:
        messages.error(request, 'Resume not found.')
        return redirect('resume_processor:resume_list')

def resume_structured(request, resume_id):
    """View for showing structured analysis of a specific resume."""
    try:
        resume = Resume.objects.get(id=resume_id)
        
        # Load structured data if available
        structured_data = resume.parsed_data if resume.parsed_data else None
        
        return render(request, 'resume_processor/resume_structured.html', {
            'resume': resume,
            'structured_data': structured_data,
            'show_nav': True
        })
    except Resume.DoesNotExist:
        messages.error(request, 'Resume not found.')
        return redirect('resume_processor:resume_list')

def resume_json(request, resume_id):
    """View for showing raw JSON data for ML model input."""
    try:
        resume = Resume.objects.get(id=resume_id)
        
        # Load structured data if available
        structured_data = resume.parsed_data if resume.parsed_data else None
        structured_data_json = None
        if structured_data:
            try:
                import json
                from .enhanced_pdf_parser import PDFParser
                pdf_parser = PDFParser()
                cleaned_data = pdf_parser.clean_json(structured_data)
                structured_data_json = json.dumps(cleaned_data, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.error(f"Error processing structured data: {str(e)}")
        
        return render(request, 'resume_processor/resume_json.html', {
            'resume': resume,
            'structured_data': structured_data,
            'structured_data_json': structured_data_json,
            'show_nav': True
        })
    except Resume.DoesNotExist:
        messages.error(request, 'Resume not found.')
        return redirect('resume_processor:resume_list')

def download_json(request, resume_id):
    """View to download the cleaned JSON file for a resume."""
    try:
        resume = Resume.objects.get(id=resume_id)
        if resume.parsed_data:
            import json
            from .enhanced_pdf_parser import PDFParser
            pdf_parser = PDFParser()
            cleaned_data = pdf_parser.clean_json(resume.parsed_data)
            import io
            cleaned_json_bytes = io.BytesIO(json.dumps(cleaned_data, indent=2, ensure_ascii=False).encode('utf-8'))
            response = FileResponse(cleaned_json_bytes)
            response['Content-Type'] = 'application/json'
            response['Content-Disposition'] = f'attachment; filename="{resume.filename.replace(".pdf", "_structured.json")}"'
            return response
        else:
            messages.error(request, 'Resume data not found.')
            return redirect('resume_processor:resume_detail', resume_id=resume_id)
    except Resume.DoesNotExist:
        messages.error(request, 'Resume not found.')
        return redirect('resume_processor:resume_list')
    except Exception as e:
        messages.error(request, f'Error downloading JSON: {str(e)}')
        return redirect('resume_processor:resume_detail', resume_id=resume_id)

def delete_resume(request, resume_id):
    """View to delete a processed resume and its files."""
    try:
        resume = Resume.objects.get(id=resume_id)
        # Delete associated files if they exist
        if resume.original_file_path and os.path.exists(resume.original_file_path):
            os.remove(resume.original_file_path)
        # Clean up JSON file if it exists
        json_file_path = os.path.join(settings.MEDIA_ROOT, 'parsed_resumes', f"{resume.candidate_id}.json")
        if os.path.exists(json_file_path):
            os.remove(json_file_path)
        # Optionally, delete extracted text file if you save it separately
        # if resume.text_file_path and os.path.exists(resume.text_file_path):
        #     os.remove(resume.text_file_path)
        resume.delete()
        messages.success(request, 'Resume deleted successfully.')
    except Resume.DoesNotExist:
        messages.error(request, 'Resume not found.')
    except Exception as e:
        messages.error(request, f'Error deleting resume: {str(e)}')
    return redirect('resume_processor:resume_list')

@csrf_exempt
def process_resume_api(request, resume_id):
    """API endpoint to process a resume."""
    if request.method == 'POST':
        try:
            resume = Resume.objects.get(id=resume_id)
            pdf_parser = PDFParser()
            resume.processing_status = 'processing'
            resume.save()
            json_file_path = pdf_parser.extract_to_json(resume.original_file_path, resume.filename)
            # Load and store structured data
            import json
            with open(json_file_path, 'r', encoding='utf-8') as f:
                structured_data = json.load(f)
            resume.parsed_data = structured_data
            resume.processing_status = 'completed'
            resume.processed_at = timezone.now()
            resume.save()
            return JsonResponse({
                'success': True,
                'status': resume.processing_status,
                'error_message': resume.error_message
            })
        except Resume.DoesNotExist:
            return JsonResponse({'success': False, 'error': 'Resume not found'}, status=404)
        except Exception as e:
            resume.processing_status = 'failed'
            resume.error_message = str(e)
            resume.save()
            return JsonResponse({'success': False, 'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)

def ranking_view(request):
    """View for showing resume rankings and scores."""
    try:
        # Get the most recent batch results from session
        batch_results = request.session.get('batch_results', [])
        
        if not batch_results:
            # No batch results, show empty state
            context = {
                'ranked_resumes': [],
                'show_nav': True,
                'total_resumes': 0,
                'ranked_count': 0,
                'batch_info': None,
                'no_batches': True
            }
            return render(request, 'resume_processor/ranking.html', context)
        
        # Get the most recent batch
        latest_batch = batch_results[-1]
        batch_results_data = latest_batch['results']
        
        
        # Check if there was an error in processing
        if 'error' in batch_results_data:
            context = {
                'ranked_resumes': [],
                'show_nav': True,
                'total_resumes': 0,
                'ranked_count': 0,
                'batch_info': latest_batch,
                'processing_error': batch_results_data['error']
            }
            return render(request, 'resume_processor/ranking.html', context)
        
        # Process the batch results to create ranking display
        ranked_resumes = []
        resumes_data = batch_results_data.get('resumes', [])
        final_ranking = batch_results_data.get('final_ranking', [])
        
        # Create a mapping of resume ID to ranking info
        ranking_map = {r['id']: r for r in final_ranking}
        
        for resume_data in resumes_data:
            resume_id = resume_data['id']
            ranking_info = ranking_map.get(resume_id, {})
            
            # Derive a display name (prefer parsed name; fallback to prettified filename)
            meta = resume_data.get('meta', {})
            parsed = resume_data.get('parsed', {})
            source_file = meta.get('source_file', 'Unknown.pdf')
            base_name = os.path.splitext(source_file)[0]
            pretty_base = base_name.replace('_', ' ').replace('-', ' ').strip()
            pretty_base = pretty_base.title() if pretty_base else base_name

            candidate_name = None
            if isinstance(parsed, dict):
                candidate_name = parsed.get('name') or (
                    isinstance(parsed.get('profile'), dict) and parsed.get('profile', {}).get('name')
                )
            candidate_name = candidate_name or meta.get('candidate_name')
            
            # If no name found in parsed data, try to extract from resume sections
            if not candidate_name:
                sections = resume_data.get('sections', [])
                for section in sections:
                    content = section.get('content', [])
                    if content:
                        # Look for lines that look like names (2-4 capitalized words)
                        for line in content[:3]:  # Check first 3 lines of each section
                            line = line.strip()
                            if line and not line.startswith(('•', '-', '*', '1.', '2.', '3.')):
                                words = line.split()
                                if 2 <= len(words) <= 4 and all(word[0].isupper() for word in words if word and word[0].isalpha()):
                                    candidate_name = line
                                    break
                    if candidate_name:
                        break
            
            display_name = candidate_name if candidate_name else pretty_base
            
            scores_obj = resume_data.get('scores', {})
            default_score = (
                scores_obj.get('final_pre_llm_display')
                or scores_obj.get('semantic_score')
                or scores_obj.get('tfidf_section_score')
                or 0
            )

            ranking_display = {
                'resume_id': resume_id,
                'display_name': display_name,
                'candidate_name': display_name,  # Add candidate_name field
                'has_ranking': resume_id in ranking_map,
                'ranking_score': float(default_score) if default_score is not None else 0.0,
                'ranking_reason': None,
                'rank': None,
                'scores': resume_data.get('scores', {}),
                'matched_skills': resume_data.get('matched_skills', []),
                'parsed': resume_data.get('parsed', {}),
                'meta': resume_data.get('meta', {})
            }
            
            if ranking_info:
                snap = ranking_info.get('scores_snapshot', {})
                # Prefer final_pre_llm_display if present, else semantic, else keep default
                ranking_display['ranking_score'] = snap.get('final_pre_llm_display') or snap.get('semantic') or ranking_display['ranking_score']
                ranking_display['ranking_reason'] = ranking_info.get('reasoning', '')
                ranking_display['rank'] = ranking_info.get('rank', 0)
            
            ranked_resumes.append(ranking_display)
        
        # Sort by rank if available, otherwise by score
        ranked_resumes.sort(
            key=lambda x: (x['rank'] if x['rank'] is not None else 0, 
                          x['ranking_score'] if x['ranking_score'] is not None else 0),
            reverse=False  # Lower rank numbers first
        )
        
        context = {
            'ranked_resumes': ranked_resumes,
            'show_nav': True,
            'total_resumes': len(ranked_resumes),
            'ranked_count': sum(1 for r in ranked_resumes if r['has_ranking']),
            'batch_info': latest_batch,
            'no_batches': False,
            'processing_error': None
        }
        
        return render(request, 'resume_processor/ranking.html', context)
        
    except Exception as e:
        logger.error(f"Error in ranking view: {str(e)}")
        messages.error(request, f'Error loading rankings: {str(e)}')
        return redirect('resume_processor:resume_list')

def all_batches_view(request):
    """View for showing all batch processing results."""
    try:
        batch_results = request.session.get('batch_results', [])
        
        # Process batch results for display
        processed_batches = []
        for i, batch in enumerate(reversed(batch_results)):  # Show newest first
            batch_info = {
                'index': len(batch_results) - i,
                'timestamp': batch['timestamp'],
                'job_description': batch['job_description'],
                'resume_count': batch['resume_count'],
                'filenames': batch['filenames'],
                'has_error': 'error' in batch['results'],
                'error_message': batch['results'].get('error') if 'error' in batch['results'] else None,
                'top_candidates': batch['results'].get('batch_summary', {}).get('top_candidates', []) if 'error' not in batch['results'] else []
            }
            processed_batches.append(batch_info)
        
        context = {
            'batches': processed_batches,
            'show_nav': True,
            'total_batches': len(processed_batches)
        }
        
        return render(request, 'resume_processor/all_batches.html', context)
        
    except Exception as e:
        logger.error(f"Error in all batches view: {str(e)}")
        messages.error(request, f'Error loading batch history: {str(e)}')
        return redirect('resume_processor:resume_list')

@csrf_exempt
def deterministic_ranking_api(request):
    """API endpoint for deterministic resume ranking."""
    if request.method == 'POST':
        try:
            from .deterministic_ranker import process_ranking_payload
            
            # Get JSON payload from request body
            payload_json = request.body.decode('utf-8')
            
            # Process ranking
            result_json = process_ranking_payload(payload_json)
            
            # Return JSON response
            return HttpResponse(result_json, content_type='application/json')
            
        except Exception as e:
            logger.error(f"Error in deterministic ranking: {str(e)}")
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Only POST method allowed'}, status=405)
