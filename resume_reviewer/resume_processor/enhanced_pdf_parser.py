import pdfplumber
import pytesseract
from PIL import Image
import json
import os
from django.conf import settings
from django.utils import timezone
import logging
from typing import List, Dict, Any, Tuple
import re

logger = logging.getLogger(__name__)

class PDFParser:
    """Enhanced PDF parser that accurately extracts resume data with proper header detection and content grouping."""
    
    def __init__(self, output_dir: str = None):
        # Handle output directory setup
        if output_dir:
            self.text_output_dir = os.path.join(output_dir, 'extracted_text')
            self.structured_output_dir = os.path.join(output_dir, 'structured_data')
        else:
            # Try to get Django settings, fallback to current directory
            try:
                self.text_output_dir = os.path.join(settings.MEDIA_ROOT, 'extracted_text')
                self.structured_output_dir = os.path.join(settings.MEDIA_ROOT, 'structured_data')
            except (ImportError, AttributeError):
                # Fallback to current directory
                current_dir = os.getcwd()
                self.text_output_dir = os.path.join(current_dir, 'extracted_text')
                self.structured_output_dir = os.path.join(current_dir, 'structured_data')
        
        # Create directories if they don't exist
        os.makedirs(self.text_output_dir, exist_ok=True)
        os.makedirs(self.structured_output_dir, exist_ok=True)
        
        # Comprehensive list of resume section headers
        self.section_headers = {
            'contact': ['contact', 'contact information', 'personal information', 'personal details', 'address', 'phone', 'email'],
            'summary': ['summary', 'profile', 'professional summary', 'career summary', 'objective', 'career objective', 'professional profile', 'personal statement'],
            'experience': ['experience', 'work experience', 'professional experience', 'employment', 'employment history', 'career history', 'work history', 'professional background'],
            'education': ['education', 'educational background', 'academic background', 'qualifications', 'academic qualifications', 'academic history'],
            'skills': ['skills', 'technical skills', 'core competencies', 'competencies', 'expertise', 'proficiencies', 'capabilities', 'abilities'],
            'certifications': ['certifications', 'certificates', 'professional certifications', 'licenses', 'accreditations'],
            'projects': ['projects', 'key projects', 'notable projects', 'project experience', 'project work'],
            'awards': ['awards', 'honors', 'achievements', 'recognition', 'accomplishments', 'distinctions'],
            'publications': ['publications', 'research', 'papers', 'articles', 'papers published'],
            'languages': ['languages', 'language skills', 'linguistic skills', 'foreign languages'],
            'interests': ['interests', 'hobbies', 'personal interests', 'activities', 'extracurricular'],
            'references': ['references', 'professional references', 'referees', 'character references'],
            'availability': ['availability', 'available', 'availability status', 'when available'],
            'leadership': ['leadership', 'leadership roles', 'leadership experience', 'leadership positions', 'leadership activities'],
            'volunteer': ['volunteer', 'volunteer work', 'volunteer experience', 'community service'],
            'training': ['training', 'professional development', 'courses', 'workshops', 'seminars']
        }
        
        # Content to filter out (page headers, tips, etc.)
        self.filter_patterns = [
            r'^(Page\s+)?\d+(\s+of\s+\d+)?$',  # Page numbers
            r'^(Resume|CV|Curriculum\s+Vitae)$',  # Document titles
            r'^Tips?\s*:',  # Tips
            r'^Note\s*:',  # Notes
            r'^Remember\s*:',  # Reminders
            r'^Keep\s+in\s+mind\s*:',  # Instructions
            r'^supervisors?\s+after\s+completing\s+work\s+experience',  # Specific problematic content
            r'^This\s+is\s+a\s+sample',  # Sample text
            r'^For\s+best\s+results',  # Instructions
            r'^Make\s+sure\s+to',  # Instructions
            r'^Include\s+your',  # Instructions
            r'^Don\'t\s+forget\s+to',  # Instructions
        ]

    def extract_to_json(self, pdf_file_path: str, original_filename: str) -> str:
        structured_data = self._extract_structured_data(pdf_file_path)
        json_file_path = self._save_json(structured_data, original_filename)
        return json_file_path

    def _extract_structured_data(self, pdf_file_path: str) -> Dict[str, Any]:
        structured_data = {
            'success': True,
            'sections': [],
            'summary': {},
            'layout_metadata': {},  # New: Store layout information for ML processing
            'error': None
        }
        
        try:
            with pdfplumber.open(pdf_file_path) as pdf:
                # Extract text with layout metadata
                text_elements = self._extract_text_with_layout(pdf)
                
                if not text_elements:
                    structured_data['success'] = False
                    structured_data['error'] = 'No text could be extracted from PDF'
                    return structured_data
                
                # Store layout metadata for ML processing
                structured_data['layout_metadata'] = {
                    'text_elements': text_elements,
                    'font_statistics': self._calculate_font_statistics(text_elements),
                    'layout_analysis': self._analyze_layout(text_elements)
                }
                
                # Extract clean text lines for current processing
                all_lines = [elem['text'] for elem in text_elements if elem['text'].strip()]
                
                # Clean and preprocess lines
                cleaned_lines = self._clean_lines(all_lines)
                
                # Detect actual section headers and group content
                sections = self._detect_sections_and_group_content(cleaned_lines)
                
                # Apply final quality check
                sections = self._final_quality_check(sections)
                
                structured_data['sections'] = sections
                structured_data['summary'] = self._generate_accurate_summary(sections, cleaned_lines)
                
        except Exception as e:
            logger.error(f"Error extracting structured data: {str(e)}")
            structured_data['success'] = False
            structured_data['error'] = str(e)
            
        return structured_data

    def _extract_text_with_layout(self, pdf) -> List[Dict[str, Any]]:
        """Extract text with layout metadata (font size, weight, coordinates)."""
        text_elements = []
        
        for page_num, page in enumerate(pdf.pages):
            try:
                # Extract text objects with layout information
                chars = page.chars
                if not chars:
                    # Fallback to basic text extraction
                    text = page.extract_text()
                    if not text:
                        text = self._ocr_page(page)
                    if text:
                        # Create basic text elements without layout info
                        lines = text.split('\n')
                        for i, line in enumerate(lines):
                            if line.strip():
                                text_elements.append({
                                    'text': line.strip(),
                                    'page': page_num,
                                    'y': i * 20,  # Approximate position
                                    'x': 0,
                                    'font_size': 12,  # Default
                                    'font_name': 'unknown',
                                    'is_bold': False,
                                    'is_italic': False
                                })
                    continue
                
                # Group characters into text elements by proximity and formatting
                text_blocks = self._group_chars_into_blocks(chars, page_num)
                text_elements.extend(text_blocks)
                
            except Exception as e:
                logger.warning(f"Error processing page {page_num}: {str(e)}")
                continue
                
        return text_elements

    def _group_chars_into_blocks(self, chars: List[Dict], page_num: int) -> List[Dict[str, Any]]:
        """Group individual characters into text blocks based on proximity and formatting."""
        if not chars:
            return []
        
        # Sort characters by y-coordinate, then x-coordinate
        sorted_chars = sorted(chars, key=lambda c: (c['top'], c['x0']))
        
        text_blocks = []
        current_block = {
            'chars': [],
            'font_size': None,
            'font_name': None,
            'is_bold': False,
            'is_italic': False,
            'page': page_num
        }
        
        for char in sorted_chars:
            # Check if this character belongs to the current block
            if current_block['chars']:
                last_char = current_block['chars'][-1]
                
                # Check if characters are close enough to be in same line
                y_distance = abs(char['top'] - last_char['top'])
                x_distance = char['x0'] - last_char['x1']
                
                # Same line if y-distance is small and x-distance is reasonable
                # Reduced x_distance threshold for better fragment detection
                if y_distance < 5 and x_distance < 20:
                    current_block['chars'].append(char)
                else:
                    # Start new block
                    if current_block['chars']:
                        text_blocks.append(self._create_text_element(current_block))
                    current_block = {
                        'chars': [char],
                        'font_size': None,
                        'font_name': None,
                        'is_bold': False,
                        'is_italic': False,
                        'page': page_num
                    }
            else:
                current_block['chars'].append(char)
        
        # Add the last block
        if current_block['chars']:
            text_blocks.append(self._create_text_element(current_block))
        
        # CRITICAL: Post-process text blocks to fix fragmentation
        return self._reconstruct_fragmented_text(text_blocks)

    def _reconstruct_fragmented_text(self, text_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Reconstruct fragmented text, especially ordinals and bullet points."""
        if not text_blocks:
            return text_blocks
        
        reconstructed_blocks = []
        i = 0
        
        while i < len(text_blocks):
            current_block = text_blocks[i]
            current_text = current_block.get('text', '').strip()
            
            # Look ahead for fragments to reconstruct
            reconstructed_text = self._reconstruct_text_fragments(text_blocks, i)
            
            if reconstructed_text != current_text:
                # Text was reconstructed, update the block
                current_block['text'] = reconstructed_text
                # Skip the blocks that were merged
                fragments_used = self._count_fragments_used(text_blocks, i, reconstructed_text)
                i += fragments_used
            else:
                i += 1
            
            reconstructed_blocks.append(current_block)
        
        return reconstructed_blocks

    def _reconstruct_text_fragments(self, text_blocks: List[Dict[str, Any]], start_index: int) -> str:
        """Reconstruct text fragments starting from a given index."""
        if start_index >= len(text_blocks):
            return ""
        
        current_text = text_blocks[start_index].get('text', '').strip()
        
        # Pattern 1: Ordinal number reconstruction (1 + st, 2 + nd, etc.)
        if re.match(r'^\d+$', current_text):
            reconstructed = self._reconstruct_ordinal(text_blocks, start_index)
            if reconstructed != current_text:
                return reconstructed
        
        # Pattern 2: Bullet point reconstruction
        if current_text in ['st', 'nd', 'rd', 'th'] and start_index > 0:
            prev_text = text_blocks[start_index - 1].get('text', '').strip()
            if re.match(r'^\d+$', prev_text):
                # This is part of an ordinal that should have been caught earlier
                return current_text
        
        # Pattern 3: Fragmented bullet points
        if current_text in ['•', '●', '-', '*'] or re.match(r'^[•●\-\*]', current_text):
            reconstructed = self._reconstruct_bullet_point(text_blocks, start_index)
            if reconstructed != current_text:
                return reconstructed
        
        # Pattern 4: Fragmented sentences (look for continuation patterns)
        if start_index < len(text_blocks) - 1:
            next_text = text_blocks[start_index + 1].get('text', '').strip()
            if self._should_merge_fragments(current_text, next_text):
                reconstructed = self._merge_sentence_fragments(text_blocks, start_index)
                if reconstructed != current_text:
                    return reconstructed
        
        return current_text

    def _reconstruct_ordinal(self, text_blocks: List[Dict[str, Any]], start_index: int) -> str:
        """Reconstruct ordinal numbers (1st, 2nd, 3rd, etc.)."""
        if start_index >= len(text_blocks):
            return ""
        
        current_text = text_blocks[start_index].get('text', '').strip()
        
        # Check if current text is a number
        if not re.match(r'^\d+$', current_text):
            return current_text
        
        # Look for ordinal suffix in next blocks
        ordinal_suffixes = ['st', 'nd', 'rd', 'th']
        reconstructed_parts = [current_text]
        
        for i in range(start_index + 1, min(start_index + 3, len(text_blocks))):
            next_text = text_blocks[i].get('text', '').strip()
            
            if next_text in ordinal_suffixes:
                # Found ordinal suffix
                reconstructed_parts.append(next_text)
                
                # Check for additional fragments (like "and 2nd")
                if i + 1 < len(text_blocks):
                    following_text = text_blocks[i + 1].get('text', '').strip()
                    if following_text.lower() in ['and', '&', ',']:
                        reconstructed_parts.append(' ' + following_text)
                        
                        # Look for next number
                        if i + 2 < len(text_blocks):
                            next_num_text = text_blocks[i + 2].get('text', '').strip()
                            if re.match(r'^\d+$', next_num_text):
                                reconstructed_parts.append(' ' + next_num_text)
                                
                                # Look for its suffix
                                if i + 3 < len(text_blocks):
                                    next_suffix = text_blocks[i + 3].get('text', '').strip()
                                    if next_suffix in ordinal_suffixes:
                                        reconstructed_parts.append(next_suffix)
                
                break
            elif next_text and not next_text.isspace():
                # If we hit non-whitespace that's not an ordinal suffix, stop
                break
        
        return ''.join(reconstructed_parts)

    def _reconstruct_bullet_point(self, text_blocks: List[Dict[str, Any]], start_index: int) -> str:
        """Reconstruct bullet points with proper formatting."""
        if start_index >= len(text_blocks):
            return ""
        
        current_text = text_blocks[start_index].get('text', '').strip()
        
        # Standardize bullet character
        if current_text in ['•', '●', '-', '*', '▪', '▫']:
            bullet_char = '●'
        elif re.match(r'^[•●\-\*]', current_text):
            bullet_char = '●'
            # Keep any text after the bullet
            remaining_text = current_text[1:].strip()
            if remaining_text:
                return bullet_char + ' ' + remaining_text
        else:
            return current_text
        
        # Look for content following the bullet
        content_parts = []
        
        for i in range(start_index + 1, min(start_index + 5, len(text_blocks))):
            next_text = text_blocks[i].get('text', '').strip()
            
            if next_text:
                # Check if this looks like bullet content
                if not re.match(r'^[•●\-\*]', next_text):
                    content_parts.append(next_text)
                else:
                    # Hit another bullet, stop
                    break
            else:
                # Empty text, might be spacing
                continue
        
        if content_parts:
            return bullet_char + ' ' + ' '.join(content_parts)
        else:
            return bullet_char + ' '

    def _should_merge_fragments(self, current_text: str, next_text: str) -> bool:
        """Determine if two text fragments should be merged."""
        if not current_text or not next_text:
            return False
        
        # Merge if current ends with incomplete word and next starts with completion
        if (current_text[-1].isalnum() and next_text[0].isalnum() and 
            not current_text.endswith('.') and not next_text[0].isupper()):
            return True
        
        # Merge ordinal fragments
        if re.match(r'^\d+$', current_text) and next_text in ['st', 'nd', 'rd', 'th']:
            return True
        
        # Merge if current is a fragment and next continues the sentence
        if (len(current_text) < 3 and current_text.isalnum() and 
            next_text[0].islower()):
            return True
        
        return False

    def _merge_sentence_fragments(self, text_blocks: List[Dict[str, Any]], start_index: int) -> str:
        """Merge sentence fragments into coherent text."""
        if start_index >= len(text_blocks):
            return ""
        
        merged_parts = [text_blocks[start_index].get('text', '').strip()]
        
        for i in range(start_index + 1, min(start_index + 5, len(text_blocks))):
            next_text = text_blocks[i].get('text', '').strip()
            
            if next_text and self._should_merge_fragments(merged_parts[-1], next_text):
                merged_parts.append(next_text)
            else:
                break
        
        return ' '.join(merged_parts)

    def _count_fragments_used(self, text_blocks: List[Dict[str, Any]], start_index: int, reconstructed_text: str) -> int:
        """Count how many text blocks were used in reconstruction."""
        original_text = text_blocks[start_index].get('text', '').strip()
        
        if reconstructed_text == original_text:
            return 1
        
        # Count based on the complexity of reconstruction
        fragments_count = 1
        
        # Check for ordinal reconstruction
        if re.match(r'^\d+(st|nd|rd|th)', reconstructed_text):
            fragments_count = 2
            
            # Check for compound ordinals (1st and 2nd)
            if ' and ' in reconstructed_text or ' & ' in reconstructed_text:
                fragments_count = 5
        
        # Check for bullet point reconstruction
        elif reconstructed_text.startswith('● '):
            content_words = len(reconstructed_text.split()) - 1  # Minus the bullet
            fragments_count = min(content_words, 3)
        
        return fragments_count

    def _create_text_element(self, block: Dict) -> Dict[str, Any]:
        """Create a text element from a block of characters."""
        if not block['chars']:
            return {}
        
        # Combine text
        text = ''.join(char['text'] for char in block['chars'])
        
        # Calculate position (use first character's position)
        first_char = block['chars'][0]
        x = first_char['x0']
        y = first_char['top']
        
        # Determine font properties (use most common)
        font_sizes = [char.get('size', 12) for char in block['chars']]
        font_names = [char.get('fontname', 'unknown') for char in block['chars']]
        
        # Use median font size to avoid outliers
        font_sizes.sort()
        font_size = font_sizes[len(font_sizes) // 2] if font_sizes else 12
        
        # Most common font name
        font_name = max(set(font_names), key=font_names.count) if font_names else 'unknown'
        
        # Check for bold/italic (simplified detection)
        is_bold = any('bold' in fn.lower() for fn in font_names)
        is_italic = any('italic' in fn.lower() for fn in font_names)
        
        return {
            'text': text.strip(),
            'page': block['page'],
            'x': x,
            'y': y,
            'font_size': font_size,
            'font_name': font_name,
            'is_bold': is_bold,
            'is_italic': is_italic,
            'char_count': len(block['chars'])
        }

    def _calculate_font_statistics(self, text_elements: List[Dict]) -> Dict[str, Any]:
        """Calculate font statistics for ML feature extraction."""
        if not text_elements:
            return {}
        
        font_sizes = [elem.get('font_size', 12) for elem in text_elements]
        font_sizes = [size for size in font_sizes if size > 0]
        
        if not font_sizes:
            return {}
        
        return {
            'mean_font_size': sum(font_sizes) / len(font_sizes),
            'median_font_size': sorted(font_sizes)[len(font_sizes) // 2],
            'max_font_size': max(font_sizes),
            'min_font_size': min(font_sizes),
            'font_size_std': (sum((x - sum(font_sizes) / len(font_sizes)) ** 2 for x in font_sizes) / len(font_sizes)) ** 0.5,
            'bold_elements_count': sum(1 for elem in text_elements if elem.get('is_bold', False)),
            'italic_elements_count': sum(1 for elem in text_elements if elem.get('is_italic', False))
        }

    def _analyze_layout(self, text_elements: List[Dict]) -> Dict[str, Any]:
        """Analyze layout patterns for ML feature extraction."""
        if not text_elements:
            return {}
        
        # Analyze positioning patterns
        y_positions = [elem.get('y', 0) for elem in text_elements]
        x_positions = [elem.get('x', 0) for elem in text_elements]
        
        # Detect potential columns
        x_positions_sorted = sorted(x_positions)
        column_boundaries = self._detect_column_boundaries(x_positions_sorted)
        
        return {
            'total_elements': len(text_elements),
            'y_range': max(y_positions) - min(y_positions) if y_positions else 0,
            'x_range': max(x_positions) - min(x_positions) if x_positions else 0,
            'column_boundaries': column_boundaries,
            'estimated_columns': len(column_boundaries) + 1
        }

    def _detect_column_boundaries(self, x_positions: List[float]) -> List[float]:
        """Detect column boundaries using clustering of x-positions."""
        if len(x_positions) < 2:
            return []
        
        # Simple clustering: find gaps larger than average
        gaps = [x_positions[i+1] - x_positions[i] for i in range(len(x_positions)-1)]
        if not gaps:
            return []
        
        avg_gap = sum(gaps) / len(gaps)
        threshold = avg_gap * 2  # Gap must be 2x average to be a column boundary
        
        boundaries = []
        for i, gap in enumerate(gaps):
            if gap > threshold:
                boundaries.append(x_positions[i] + gap / 2)
        
        return boundaries

    def _extract_all_lines(self, pdf) -> List[str]:
        """Extract all lines from PDF with OCR fallback."""
        all_lines = []
        
        for page_num, page in enumerate(pdf.pages):
            try:
                # Try text extraction first
                text = page.extract_text()
                if not text or text.strip() == '':
                    # OCR fallback for image-based PDFs
                    text = self._ocr_page(page)
                
                if text:
                    # Handle multi-column layouts
                    lines = self._handle_columns(page, text)
                    all_lines.extend(lines)
                    
            except Exception as e:
                logger.warning(f"Error processing page {page_num}: {str(e)}")
                continue
                
        return all_lines

    def _handle_columns(self, page, text: str) -> List[str]:
        """Handle multi-column layouts by detecting and merging columns properly."""
        lines = text.split('\n')
        
        # Try to detect if this is a multi-column layout
        if page.width > 700:  # Wide page, might have columns
            try:
                # Split page into left and right halves
                mid_point = page.width / 2
                left_bbox = (0, 0, mid_point, page.height)
                right_bbox = (mid_point, 0, page.width, page.height)
                
                left_text = page.within_bbox(left_bbox).extract_text() or ""
                right_text = page.within_bbox(right_bbox).extract_text() or ""
                
                # If both columns have substantial content, merge them properly
                if len(left_text.strip()) > 50 and len(right_text.strip()) > 50:
                    left_lines = [line.strip() for line in left_text.split('\n') if line.strip()]
                    right_lines = [line.strip() for line in right_text.split('\n') if line.strip()]
                    
                    # Merge columns by interleaving based on vertical position
                    merged_lines = []
                    merged_lines.extend(left_lines)
                    merged_lines.extend(right_lines)
                    return merged_lines
                    
            except Exception as e:
                logger.warning(f"Column detection failed: {str(e)}")
        
        return [line.strip() for line in lines if line.strip()]

    def _clean_lines(self, lines: List[str]) -> List[str]:
        """Clean and filter lines, removing noise and page artifacts."""
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Skip filtered patterns
            if any(re.match(pattern, line, re.IGNORECASE) for pattern in self.filter_patterns):
                continue
                
            # Skip very short lines that are likely artifacts (but keep bullet points)
            if len(line) < 2 and not re.match(r'^[•\-\*]', line):
                continue
                
            cleaned_lines.append(line)
            
        return cleaned_lines

    def _split_sections_on_embedded_headers(self, sections):
        """Split sections if a new header is found within the content."""
        new_sections = []
        for section in sections:
            content = section['content']
            current_header = section['header']
            current_content = []
            for line in content:
                # Use header detection logic; line_index and all_lines are not meaningful here, so pass dummy values
                if self._is_actual_section_header(line, 0, [line]):
                    if current_content:
                        new_sections.append({'header': current_header, 'content': current_content})
                    current_header = line
                    current_content = []
                else:
                    current_content.append(line)
            if current_content:
                new_sections.append({'header': current_header, 'content': current_content})
        # Recursively apply until no more splits
        if len(new_sections) == len(sections):
            return new_sections
        else:
            return self._split_sections_on_embedded_headers(new_sections)

    def _detect_sections_and_group_content(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Detect actual section headers and group content properly."""
        sections = []
        
        # First pass: identify potential headers with improved detection
        potential_headers = []
        for i, line in enumerate(lines):
            if self._is_actual_section_header(line, i, lines):
                potential_headers.append((i, line))
        
        # If no clear headers found, create a single section
        if not potential_headers:
            # Look for contact info at the beginning
            contact_lines = []
            content_lines = []
            
            for i, line in enumerate(lines[:10]):  # Check first 10 lines for contact info
                if self._contains_contact_info(line):
                    contact_lines.append(line)
                else:
                    content_lines.append(line)
            
            # Add remaining lines to content
            content_lines.extend(lines[len(contact_lines):])
            
            if contact_lines:
                sections.append({
                    'header': 'Contact Information',
                    'content': contact_lines
                })
            
            if content_lines:
                sections.append({
                    'header': 'Resume Content',
                    'content': content_lines
                })
                
            return sections
        
        # Process sections based on detected headers
        for i, (header_index, header_text) in enumerate(potential_headers):
            # Determine content range for this section
            next_header_index = potential_headers[i + 1][0] if i + 1 < len(potential_headers) else len(lines)
            
            # Collect content lines for this section
            content_lines = []
            for j in range(header_index + 1, next_header_index):
                if j < len(lines):
                    content_line = lines[j].strip()
                    if content_line:
                        content_lines.append(content_line)
            
            # Group content properly
            grouped_content = self._group_content_lines(content_lines)
            
            if grouped_content:  # Only add sections with content
                sections.append({
                    'header': header_text.strip(),
                    'content': grouped_content
                })
        
        # At the end, add this post-processing step:
        sections = self._split_sections_on_embedded_headers(sections)
        return sections

    def _is_actual_section_header(self, line: str, line_index: int, all_lines: List[str]) -> bool:
        line_clean = line.strip()
        line_lower = line_clean.lower()

        # Skip empty lines
        if not line_clean:
            return False

        # --- NEW: Negative heuristics for common false headers ---
        # 1. Date or date range
        if re.search(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\b', line_clean):
            return False
        if re.search(r'\b\d{4}\b', line_clean):
            return False
        if re.search(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', line_clean):
            return False
        if re.search(r'[–\-]| to |present', line_lower):
            # Matches en dash, hyphen, 'to', or 'present' (date range indicators)
            return False

        # 2. Location pattern (e.g., "Davis, CA")
        if re.search(r',\s*[A-Za-z]{2,}(\s|$)', line_clean):
            return False

        # 3. Common job titles (expand as needed)
        job_titles = [
            'ambassador', 'engineer', 'manager', 'developer', 'analyst', 'consultant', 'intern', 'assistant',
            'officer', 'director', 'coordinator', 'specialist', 'ambassador', 'representative', 'associate', 'advisor', 'lead', 'student ambassador'
        ]
        if any(title in line_lower for title in job_titles):
            return False

        # Skip lines that are clearly content
        if self._is_clearly_content(line_clean):
            return False
        
        # Check against comprehensive section headers
        for category, headers in self.section_headers.items():
            for header in headers:
                if line_lower == header or line_lower.startswith(header + ' '):
                    return True
        
        # Additional heuristics for section headers
        score = 0
        
        # All caps and reasonable length
        if line_clean.isupper() and 2 <= len(line_clean.split()) <= 5:
            score += 3
            
        # Ends with colon
        if line_clean.endswith(':'):
            score += 2
            
        # Short line (likely a header)
        if len(line_clean.split()) <= 4 and len(line_clean) < 40:
            score += 1
            
        # Contains common header words
        header_words = ['experience', 'education', 'skills', 'summary', 'profile', 'objective', 'qualifications', 'certifications', 'projects', 'awards', 'languages', 'interests', 'references', 'availability', 'leadership', 'volunteer', 'training']
        if any(word in line_lower for word in header_words):
            score += 2
            
        # Not a bullet point or numbered item
        if not re.match(r'^[•\-\*\d]', line_clean):
            score += 1
            
        # Check if next lines look like content (supporting evidence)
        if line_index + 1 < len(all_lines):
            next_line = all_lines[line_index + 1].strip()
            if next_line and (next_line.startswith('•') or next_line.startswith('-') or len(next_line) > 30):
                score += 1
        
        return score >= 4

    def _is_clearly_content(self, line: str) -> bool:
        """Determine if a line is clearly content (not a header)."""
        # Bullet points
        if re.match(r'^[•\-\*]', line):
            return True
            
        # Numbered items
        if re.match(r'^\d+\.', line):
            return True
            
        # Long lines are likely content
        if len(line) > 80:
            return True
            
        # Contains dates
        if re.search(r'\b\d{4}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b', line):
            return True
            
        # Contains email or phone
        if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', line) or re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', line):
            return True
            
        # Contains common content indicators
        content_indicators = ['responsible for', 'managed', 'developed', 'created', 'implemented', 'achieved', 'improved', 'led', 'supervised', 'coordinated', 'worked', 'assisted', 'helped', 'supported']
        if any(indicator in line.lower() for indicator in content_indicators):
            return True
            
        # Contains specific problematic content
        problematic_content = ['supervisors after completing work experience', 'tips', 'note', 'remember', 'keep in mind']
        if any(content in line.lower() for content in problematic_content):
            return True
            
        return False

    def _group_content_lines(self, content_lines: List[str]) -> List[str]:
        """Group content lines properly, keeping bullet points and related text together."""
        if not content_lines:
            return []
        
        # First, apply text reconstruction to fix fragmentation
        reconstructed_lines = self._apply_line_level_reconstruction(content_lines)
        
        grouped_content = []
        current_item = ""
        
        for line in reconstructed_lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this is a bullet point or numbered item
            if re.match(r'^[•●\-\*]', line) or re.match(r'^\d+\.', line):
                # Save previous item if exists
                if current_item:
                    # Validate and clean the item before adding
                    cleaned_item = self._validate_and_clean_text(current_item.strip())
                    if cleaned_item:
                        grouped_content.append(cleaned_item)
                # Start new item
                current_item = line
            else:
                # This is a continuation of the current item or a standalone line
                if current_item:
                    # If it looks like a continuation (not a new sentence), append to current item
                    if not line[0].isupper() or len(line.split()) < 3:
                        current_item += " " + line
                    else:
                        # This looks like a new item, save current and start new
                        cleaned_item = self._validate_and_clean_text(current_item.strip())
                        if cleaned_item:
                            grouped_content.append(cleaned_item)
                        current_item = line
                else:
                    # No current item, this is a standalone line
                    current_item = line
        
        # Add the last item
        if current_item:
            cleaned_item = self._validate_and_clean_text(current_item.strip())
            if cleaned_item:
                grouped_content.append(cleaned_item)
        
        return grouped_content

    def _apply_line_level_reconstruction(self, lines: List[str]) -> List[str]:
        """Apply text reconstruction at the line level to fix common fragmentation issues."""
        if not lines:
            return lines
        
        reconstructed_lines = []
        i = 0
        
        while i < len(lines):
            current_line = lines[i].strip()
            
            # Check for ordinal reconstruction across lines
            if re.match(r'^\d+$', current_line) and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line in ['st', 'nd', 'rd', 'th']:
                    # Reconstruct ordinal
                    reconstructed_line = current_line + next_line
                    
                    # Check for continuation (like "and 2nd")
                    if i + 2 < len(lines) and lines[i + 2].strip().lower() in ['and', '&']:
                        reconstructed_line += ' ' + lines[i + 2].strip()
                        if i + 3 < len(lines) and re.match(r'^\d+$', lines[i + 3].strip()):
                            reconstructed_line += ' ' + lines[i + 3].strip()
                            if i + 4 < len(lines) and lines[i + 4].strip() in ['st', 'nd', 'rd', 'th']:
                                reconstructed_line += lines[i + 4].strip()
                                i += 5
                            else:
                                i += 4
                        else:
                            i += 3
                    else:
                        i += 2
                    
                    reconstructed_lines.append(reconstructed_line)
                    continue
            
            # Check for bullet point reconstruction
            if current_line in ['•', '●', '-', '*']:
                bullet_content = '● '
                if i + 1 < len(lines):
                    bullet_content += lines[i + 1].strip()
                    i += 2
                else:
                    i += 1
                reconstructed_lines.append(bullet_content)
                continue
            
            # Check for fragmented sentences
            if (i + 1 < len(lines) and 
                len(current_line) < 50 and 
                not current_line.endswith('.') and 
                not lines[i + 1].strip().startswith('•')):
                
                # Merge with next line if it seems to be a continuation
                next_line = lines[i + 1].strip()
                if next_line and not next_line[0].isupper():
                    reconstructed_lines.append(current_line + ' ' + next_line)
                    i += 2
                    continue
            
            reconstructed_lines.append(current_line)
            i += 1
        
        return reconstructed_lines

    def _validate_and_clean_text(self, text: str) -> str:
        """Validate and clean text to ensure quality and fix common issues."""
        if not text:
            return ""
        
        # Step 1: Fix corrupted ordinals
        text = self._fix_corrupted_ordinals(text)
        
        # Step 2: Fix bullet point formatting
        text = self._fix_bullet_formatting(text)
        
        # Step 3: Fix spacing issues
        text = self._fix_spacing_issues(text)
        
        # Step 4: Remove nonsensical character combinations
        text = self._remove_nonsensical_combinations(text)
        
        # Step 5: Validate sentence coherence
        if not self._is_coherent_text(text):
            return ""
        
        return text.strip()

    def _fix_corrupted_ordinals(self, text: str) -> str:
        """Fix corrupted ordinal numbers in text."""
        # Fix patterns like "1stnd" or "2ndrd" or "stnd●"
        text = re.sub(r'(\d+)(st|nd|rd|th)(nd|rd|th|st)', r'\1\2', text)
        
        # Fix standalone ordinal fragments
        text = re.sub(r'\b(st|nd|rd|th)(nd|rd|th|st)\b', r'\1', text)
        
        # Fix ordinals that got corrupted with symbols
        text = re.sub(r'(st|nd|rd|th)[●•\-\*]+', r'\1', text)
        
        # Fix number + ordinal separated by space
        text = re.sub(r'(\d+)\s+(st|nd|rd|th)\b', r'\1\2', text)
        
        return text

    def _fix_bullet_formatting(self, text: str) -> str:
        """Fix bullet point formatting issues."""
        # Standardize bullet characters
        text = re.sub(r'^[•▪▫\-\*]+\s*', '● ', text)
        
        # Fix multiple bullets
        text = re.sub(r'●\s*●\s*', '● ', text)
        
        # Ensure space after bullet
        text = re.sub(r'^●([^\s])', r'● \1', text)
        
        # Remove corrupted bullet combinations
        text = re.sub(r'(st|nd|rd|th)●', r'\1 ●', text)
        
        return text

    def _fix_spacing_issues(self, text: str) -> str:
        """Fix spacing issues in text."""
        # Fix multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([,.;:!?])', r'\1', text)
        text = re.sub(r'([,.;:!?])\s*([a-zA-Z])', r'\1 \2', text)
        
        # Fix spacing around parentheses
        text = re.sub(r'\s*\(\s*', ' (', text)
        text = re.sub(r'\s*\)\s*', ') ', text)
        
        return text.strip()

    def _remove_nonsensical_combinations(self, text: str) -> str:
        """Remove nonsensical character combinations that indicate corruption."""
        # Remove patterns like "stnd●" or "rdth●"
        text = re.sub(r'(st|nd|rd|th)(nd|rd|th|st)[●•\-\*]*', r'\1', text)
        
        # Remove standalone fragments that don't make sense
        text = re.sub(r'\b(st|nd|rd|th)\b(?!\s+\w)', '', text)
        
        # Remove corrupted bullet combinations
        text = re.sub(r'[●•\-\*]{2,}', '●', text)
        
        return text.strip()

    def _is_coherent_text(self, text: str) -> bool:
        """Check if text is coherent and meaningful."""
        if not text or len(text.strip()) < 2:
            return False
        
        # Check for corrupted patterns
        corrupted_patterns = [
            r'(st|nd|rd|th){2,}',  # Multiple ordinal suffixes
            r'[●•\-\*]{3,}',       # Multiple bullets
            r'\b(st|nd|rd|th)\b$', # Standalone ordinal suffix
            r'^[●•\-\*]+$'         # Only bullets
        ]
        
        for pattern in corrupted_patterns:
            if re.search(pattern, text):
                return False
        
        # Check for minimum word content
        words = re.findall(r'\b\w+\b', text)
        if len(words) < 1:
            return False
        
        return True

    def _final_quality_check(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Perform final quality check on all sections."""
        quality_checked_sections = []
        
        for section in sections:
            cleaned_content = []
            
            for content_item in section.get('content', []):
                # Apply final validation
                cleaned_item = self._validate_and_clean_text(content_item)
                
                if cleaned_item and self._is_coherent_text(cleaned_item):
                    cleaned_content.append(cleaned_item)
            
            # Only add section if it has valid content
            if cleaned_content:
                quality_checked_sections.append({
                    'header': section['header'],
                    'content': cleaned_content
                })
        
        return quality_checked_sections

    def _contains_contact_info(self, line: str) -> bool:
        """Check if a line contains contact information."""
        line_lower = line.lower()
        
        # Email pattern
        if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', line):
            return True
            
        # Phone pattern
        if re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', line):
            return True
            
        # Address indicators
        address_indicators = ['street', 'avenue', 'road', 'drive', 'lane', 'blvd', 'ave', 'st', 'rd', 'dr']
        if any(indicator in line_lower for indicator in address_indicators):
            return True
            
        # City, State ZIP pattern
        if re.search(r'\b[A-Za-z\s]+,\s*[A-Za-z]{2}\s*\d{5}\b', line):
            return True
            
        return False

    def _generate_accurate_summary(self, sections: List[Dict[str, Any]], all_lines: List[str]) -> Dict[str, Any]:
        """Generate accurate summary metadata by examining actual content."""
        section_headers = [section['header'].lower() for section in sections]
        all_content = ' '.join([' '.join(section['content']) for section in sections]).lower()
        
        # Check for contact information
        has_contact_info = False
        for line in all_lines[:15]:  # Check first 15 lines
            if (re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', line) or 
                re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', line) or
                any(indicator in line.lower() for indicator in ['street', 'avenue', 'road', 'drive', 'phone', 'email', 'address'])):
                has_contact_info = True
                break
        
        # Check for education
        has_education = False
        education_indicators = ['degree', 'university', 'college', 'school', 'bachelor', 'master', 'phd', 'diploma', 'certification', 'graduated']
        if (any('education' in header for header in section_headers) or
            any(indicator in all_content for indicator in education_indicators)):
            has_education = True
        
        # Check for experience
        has_experience = False
        experience_indicators = ['experience', 'employment', 'work', 'position', 'job', 'company', 'employer', 'responsible for', 'managed', 'developed']
        if (any(indicator in ' '.join(section_headers) for indicator in ['experience', 'employment', 'work', 'career']) or
            any(indicator in all_content for indicator in experience_indicators)):
            has_experience = True
        
        # Check for skills
        has_skills = False
        skills_indicators = ['skills', 'competencies', 'proficient', 'expertise', 'technical', 'programming', 'software']
        if (any('skill' in header for header in section_headers) or
            any(indicator in all_content for indicator in skills_indicators)):
            has_skills = True
        
        return {
            'total_sections': len(sections),
            'section_names': [section['header'] for section in sections],
            'has_contact_info': has_contact_info,
            'has_education': has_education,
            'has_experience': has_experience,
            'has_skills': has_skills
        }

    def _ocr_page(self, page) -> str:
        """OCR fallback for image-based PDFs."""
        try:
            pil_img = page.to_image(resolution=300).original
            text = pytesseract.image_to_string(pil_img)
            return text
        except Exception as e:
            logger.error(f"OCR failed: {str(e)}")
            return ""

    def _save_json(self, structured_data: Dict[str, Any], original_filename: str) -> str:
        """Save structured data to JSON file."""
        base_name = os.path.splitext(os.path.basename(original_filename))[0]
        json_filename = f"{base_name}_structured.json"
        json_file_path = os.path.join(self.structured_output_dir, json_filename)
        
        with open(json_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(structured_data, json_file, indent=2, ensure_ascii=False)
            
        return json_file_path

    def extract_plain_text(self, structured_data: Dict[str, Any]) -> str:
        """Extract plain text from structured data."""
        plain_text = ""
        for section in structured_data.get('sections', []):
            plain_text += f"\n--- {section['header']} ---\n"
            for line in section['content']:
                plain_text += line + "\n"
        return plain_text.strip()

    def clean_json(self, data):
        """Clean JSON data, replacing null values with empty strings."""
        if isinstance(data, dict):
            return {k: self.clean_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.clean_json(item) for item in data]
        elif data is None:
            return ""
        else:
            return data

    def add_classification_results(self, json_file_path: str, classification_results: Dict[str, Any]):
        """Add classification results to existing JSON file."""
        with open(json_file_path, 'r+', encoding='utf-8') as f:
            data = json.load(f)
            data['classification'] = classification_results
            f.seek(0)
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.truncate() 