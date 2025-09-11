import json
import logging
from typing import List, Dict, Any, Optional
import requests
from dataclasses import dataclass
import time
import re
import os
from contextlib import contextmanager
from .config import get_llm_config

logger = logging.getLogger(__name__)

# --- Timing configuration ---
TIMING_ENABLED = os.getenv("TIMING", "0") in {"1", "true", "True"}

@contextmanager
def time_phase(name: str, bucket: Optional[Dict[str, float]] = None):
    """Context manager for timing phases."""
    if not TIMING_ENABLED:
        yield
        return
    
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info(f"[TIMING] {name}: {elapsed:.3f}s")
        if bucket is not None:
            bucket[name] = elapsed

@dataclass
class RankingResult:
    """Data class for ranking results."""
    candidate_id: str
    rank: int
    score: float
    reasoning: str
    strengths: List[str]
    gaps: List[str]
    confidence: float

class LLMRanker:
    """LLM-based ranking system for resume evaluation (OpenAI or Google Gemini)."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        cfg = get_llm_config()
        # Allow explicit api_key/model override from caller; otherwise use config
        self.provider = cfg.get('provider', 'openai')
        self.api_key = api_key or cfg.get('api_key')
        self.model = model or cfg.get('model')
        self.enabled = bool(self.api_key)
        # Endpoints
        self.base_url_openai = "https://api.openai.com/v1/chat/completions"
        # Gemini REST: https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent
        self.base_url_gemini = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        
        # Blind JSON-only ranking prompt template
        self.ranking_prompt_template = """
You are an expert technical recruiter. Evaluate the candidate AGAINST THE JOB DESCRIPTION, using only evidence in the resume text.
Return ONLY valid JSON:
{
  "blind_score": 0-100,
  "evidence": ["exact short quotes from resume that justify the score"],
  "must_have_violations": ["list missing must-haves"],
  "notes": "one brief sentence"
}
Rules:
- Score must be justified by quotes.
- Penalize missing must-haves heavily.
- Do not assume skills that are not explicitly mentioned.
"""
    
    def rank_candidate(self, job_description: str, resume_data: Dict[str, Any], 
                      computed_scores: Dict[str, Any]) -> RankingResult:
        """Rank a single candidate against a job description."""
        try:
            # Prepare the prompt
            prompt = self.ranking_prompt_template.format(
                job_description=job_description,
                resume_data=json.dumps(resume_data, indent=2),
                computed_scores=json.dumps(computed_scores, indent=2)
            )
            
            # Call LLM API (strict JSON, temperature 0)
            response = self._call_llm_api(prompt)
            
            if response:
                # Parse the response
                ranking_data = self._parse_llm_response(response, prompt)
                
                return RankingResult(
                    candidate_id=resume_data.get('candidate_id', 'unknown'),
                    rank=0,  # Will be set when ranking multiple candidates
                    score=ranking_data.get('blind_score', 0),
                    reasoning='; '.join(ranking_data.get('evidence', [])),
                    strengths=[],
                    gaps=ranking_data.get('must_have_violations', []),
                    confidence=0
                )
            else:
                # Fallback ranking based on computed scores
                return self._fallback_ranking(resume_data, computed_scores)
                
        except Exception as e:
            logger.error(f"Error ranking candidate: {str(e)}")
            return self._fallback_ranking(resume_data, computed_scores)
    
    def rank_candidates(self, job_description: str, candidates_data: List[Dict[str, Any]]) -> List[RankingResult]:
        """Rank multiple candidates against a job description."""
        timing_bucket = {}
        ranking_results = []
        
        with time_phase("rank", timing_bucket):
            for candidate_data in candidates_data:
                try:
                    # Extract computed scores from candidate data
                    computed_scores = candidate_data.get('computed_scores', {})
                    
                    # Rank the candidate
                    result = self.rank_candidate(job_description, candidate_data, computed_scores)
                    ranking_results.append(result)
                    
                    # Add delay to avoid rate limiting
                    time.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error ranking candidate {candidate_data.get('candidate_id', 'unknown')}: {str(e)}")
                    continue
            
            # Sort by score (highest first) and assign ranks
            ranking_results.sort(key=lambda x: x.score, reverse=True)
            for i, result in enumerate(ranking_results):
                result.rank = i + 1
        
        # Log LLM summary
        if TIMING_ENABLED and timing_bucket:
            total_time = sum(timing_bucket.values())
            logger.info(f"[TIMING] LLM {len(candidates_data)} candidates: {total_time:.3f}s total")
        
        return ranking_results
    
    def _call_llm_api(self, prompt: str) -> Optional[str]:
        """Call the configured LLM API with the given prompt (OpenAI or Gemini)."""
        if not self.enabled:
            logger.warning("No API key provided for LLM ranking")
            return None

        timing_bucket = {}
        
        try:
            with time_phase("serialize_payload", timing_bucket):
                if self.provider == 'google':
                    # Gemini generateContent API requires role: user|model and parts
                    url = self.base_url_gemini
                    headers = {
                        "Content-Type": "application/json"
                    }
                    system_preamble = "You are a strict evaluator. Respond with ONLY valid JSON that matches the requested schema. No additional text."
                    body = {
                        "contents": [
                            {
                                "role": "user",
                                "parts": [
                                    {"text": system_preamble + "\n\n" + prompt}
                                ]
                            }
                        ],
                        "generationConfig": {"temperature": 0, "maxOutputTokens": 900}
                    }
                    params = {"key": self.api_key}
                else:
                    # OpenAI fallback
                    url = self.base_url_openai
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    }
                    body = {
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": "You are a strict evaluator. Respond with ONLY valid JSON that matches the requested schema. No additional text."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0,
                        "max_tokens": 900
                    }
                    params = None
            
            with time_phase("http_post", timing_bucket):
                if self.provider == 'google':
                    resp = requests.post(url, headers=headers, params=params, json=body, timeout=10)
                else:
                    resp = requests.post(url, headers=headers, json=body, timeout=10)
            
            with time_phase("parse_response", timing_bucket):
                if resp.status_code == 200:
                    if self.provider == 'google':
                        data = resp.json()
                        # Extract text from candidates → content → parts
                        try:
                            return data["candidates"][0]["content"]["parts"][0]["text"]
                        except Exception:
                            logger.error(f"Unexpected Gemini response format: {data}")
                            return None
                    else:
                        result = resp.json()
                        return result['choices'][0]['message']['content']
                else:
                    if self.provider == 'google':
                        logger.error(f"Gemini API error: {resp.status_code} - {resp.text}")
                    else:
                        logger.error(f"OpenAI API error: {resp.status_code} - {resp.text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error calling LLM API: {str(e)}")
            return None
    
    def _parse_llm_response(self, response: str, prompt: str) -> Dict[str, Any]:
        """Strict JSON parse; if it fails, retry API once with a short reminder."""
        def strip_code_fences(text: str) -> str:
            t = text.strip()
            if t.startswith("```"):
                # remove leading fence
                t = t.split("\n", 1)[1] if "\n" in t else t
            if t.endswith("```"):
                t = t.rsplit("```", 1)[0]
            return t.strip()

        def extract_braced_json(text: str) -> Optional[str]:
            start = text.find('{')
            if start == -1:
                return None
            depth = 0
            for i in range(start, len(text)):
                ch = text[i]
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        return text[start:i+1]
            return None

        def remove_trailing_commas(s: str) -> str:
            import re
            # Remove trailing commas before } or ]
            s = re.sub(r',\s*([}\]])', r'\1', s)
            return s

        try:
            return json.loads(response)
        except Exception:
            # Try fences/substring extraction first before retrying API
            cleaned = strip_code_fences(response)
            candidate = extract_braced_json(cleaned) or cleaned
            try:
                return json.loads(remove_trailing_commas(candidate))
            except Exception as e:
                logger.warning(f"Strict JSON parse failed: {e}; retrying once with reminder")
                reminder = "Return ONLY JSON that matches the schema."
                retry_resp = self._call_llm_api(reminder + "\n\n" + prompt)
                if not retry_resp:
                    return self._fallback_parse_response(response)
                cleaned2 = strip_code_fences(retry_resp)
                candidate2 = extract_braced_json(cleaned2) or cleaned2
                try:
                    return json.loads(remove_trailing_commas(candidate2))
                except Exception as e2:
                    logger.warning(f"Retry strict parse failed: {e2}")
                    return self._fallback_parse_response(retry_resp)
    
    def _fallback_parse_response(self, response: str) -> Dict[str, Any]:
        """Fallback parsing for non-JSON responses."""
        # Simple keyword-based parsing
        response_lower = response.lower()
        
        # Extract score
        score = 50  # Default score
        if 'score' in response_lower:
            try:
                score_match = re.search(r'score[:\s]*(\d+)', response_lower)
                if score_match:
                    score = int(score_match.group(1))
            except:
                pass
        
        # Extract confidence
        confidence = 70  # Default confidence
        if 'confidence' in response_lower:
            try:
                conf_match = re.search(r'confidence[:\s]*(\d+)', response_lower)
                if conf_match:
                    confidence = int(conf_match.group(1))
            except:
                pass
        
        return {
            'score': score,
            'reasoning': response[:500] + '...' if len(response) > 500 else response,
            'strengths': ['Strengths could not be parsed from response'],
            'gaps': ['Gaps could not be parsed from response'],
            'confidence': confidence
        }

    
    def _fallback_ranking(self, resume_data: Dict[str, Any], computed_scores: Dict[str, Any]) -> RankingResult:
        """Fallback ranking when LLM is not available."""
        # Calculate a simple score based on computed metrics
        tfidf_score = computed_scores.get('tfidf_score', 0)
        skill_diversity = computed_scores.get('skill_diversity', 0)
        semantic_richness = computed_scores.get('semantic_richness', 0)
        
        # Normalize and combine scores
        normalized_tfidf = min(tfidf_score / 10, 1.0)  # Normalize to 0-1
        normalized_skills = min(skill_diversity / 20, 1.0)  # Normalize to 0-1
        normalized_semantic = min(semantic_richness / 10, 1.0)  # Normalize to 0-1
        
        # Weighted average
        fallback_score = (normalized_tfidf * 0.4 + normalized_skills * 0.4 + normalized_semantic * 0.2) * 100
        
        return RankingResult(
            candidate_id=resume_data.get('candidate_id', 'unknown'),
            rank=0,
            score=fallback_score,
            reasoning="Fallback ranking based on computed metrics (LLM not available)",
            strengths=["Automated scoring based on content analysis"],
            gaps=["Detailed reasoning not available"],
            confidence=60
        )


class RankingManager:
    """Manager for handling ranking operations and results."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.llm_ranker = LLMRanker(api_key)
        self.ranking_history = []
    
    def create_ranking_session(self, job_description: str, candidates_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a new ranking session for a job posting."""
        session_id = f"ranking_{int(time.time())}"
        
        # Rank all candidates
        ranking_results = self.llm_ranker.rank_candidates(job_description, candidates_data)
        
        # Create session data
        session_data = {
            'session_id': session_id,
            'job_description': job_description,
            'candidates_count': len(candidates_data),
            'ranking_results': [
                {
                    'candidate_id': result.candidate_id,
                    'rank': result.rank,
                    'score': result.score,
                    'reasoning': result.reasoning,
                    'strengths': result.strengths,
                    'gaps': result.gaps,
                    'confidence': result.confidence
                }
                for result in ranking_results
            ],
            'created_at': time.time(),
            'status': 'completed'
        }
        
        # Store in history
        self.ranking_history.append(session_data)
        
        return session_data
    
    def get_ranking_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get a summary of ranking results for a session."""
        session = next((s for s in self.ranking_history if s['session_id'] == session_id), None)
        
        if not session:
            return None
        
        results = session['ranking_results']
        
        summary = {
            'session_id': session_id,
            'total_candidates': len(results),
            'average_score': sum(r['score'] for r in results) / len(results) if results else 0,
            'top_candidates': results[:5],  # Top 5 candidates
            'score_distribution': self._calculate_score_distribution(results),
            'created_at': session['created_at']
        }
        
        return summary
    
    def _calculate_score_distribution(self, results: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate score distribution across ranges."""
        distribution = {
            'excellent (90-100)': 0,
            'good (80-89)': 0,
            'average (70-79)': 0,
            'below_average (60-69)': 0,
            'poor (0-59)': 0
        }
        
        for result in results:
            score = result['score']
            if score >= 90:
                distribution['excellent (90-100)'] += 1
            elif score >= 80:
                distribution['good (80-89)'] += 1
            elif score >= 70:
                distribution['average (70-79)'] += 1
            elif score >= 60:
                distribution['below_average (60-69)'] += 1
            else:
                distribution['poor (0-59)'] += 1
        
        return distribution
    
    def export_ranking_results(self, session_id: str, format: str = 'json') -> Optional[str]:
        """Export ranking results in specified format."""
        session = next((s for s in self.ranking_history if s['session_id'] == session_id), None)
        
        if not session:
            return None
        
        if format.lower() == 'json':
            return json.dumps(session, indent=2)
        elif format.lower() == 'csv':
            return self._export_to_csv(session)
        else:
            logger.error(f"Unsupported export format: {format}")
            return None
    
    def _export_to_csv(self, session: Dict[str, Any]) -> str:
        """Export ranking results to CSV format."""
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(['Rank', 'Candidate ID', 'Score', 'Confidence', 'Reasoning'])
        
        # Write data
        for result in session['ranking_results']:
            writer.writerow([
                result['rank'],
                result['candidate_id'],
                result['score'],
                result['confidence'],
                result['reasoning'][:100] + '...' if len(result['reasoning']) > 100 else result['reasoning']
            ])
        
        return output.getvalue() 