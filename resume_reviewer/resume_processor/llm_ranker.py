import json
import logging
from typing import List, Dict, Any, Optional
import requests
from dataclasses import dataclass
import time
import re

logger = logging.getLogger(__name__)

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
    """LLM-based ranking system for resume evaluation."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.openai.com/v1/chat/completions"
        
        # Ranking prompt template
        self.ranking_prompt_template = """
You are an expert HR recruiter evaluating candidates for a job position. You will be provided with:

1. Job Description: {job_description}
2. Candidate Resume Data: {resume_data}
3. Computed Scores: {computed_scores}

Please evaluate the candidate and provide:
1. A ranking score (0-100)
2. Detailed reasoning for your evaluation
3. Key strengths that match the job requirements
4. Potential gaps or areas of concern
5. Confidence level in your assessment (0-100)

Focus on:
- Skills alignment with job requirements
- Experience relevance and depth
- Education and certifications
- Overall fit for the role

Respond in the following JSON format:
{{
    "score": <ranking_score>,
    "reasoning": "<detailed_reasoning>",
    "strengths": ["<strength1>", "<strength2>", ...],
    "gaps": ["<gap1>", "<gap2>", ...],
    "confidence": <confidence_level>
}}
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
            
            # Call LLM API
            response = self._call_llm_api(prompt)
            
            if response:
                # Parse the response
                ranking_data = self._parse_llm_response(response)
                
                return RankingResult(
                    candidate_id=resume_data.get('candidate_id', 'unknown'),
                    rank=0,  # Will be set when ranking multiple candidates
                    score=ranking_data.get('score', 0),
                    reasoning=ranking_data.get('reasoning', ''),
                    strengths=ranking_data.get('strengths', []),
                    gaps=ranking_data.get('gaps', []),
                    confidence=ranking_data.get('confidence', 0)
                )
            else:
                # Fallback ranking based on computed scores
                return self._fallback_ranking(resume_data, computed_scores)
                
        except Exception as e:
            logger.error(f"Error ranking candidate: {str(e)}")
            return self._fallback_ranking(resume_data, computed_scores)
    
    def rank_candidates(self, job_description: str, candidates_data: List[Dict[str, Any]]) -> List[RankingResult]:
        """Rank multiple candidates against a job description."""
        ranking_results = []
        
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
        
        return ranking_results
    
    def _call_llm_api(self, prompt: str) -> Optional[str]:
        """Call the LLM API with the given prompt."""
        if not self.api_key:
            logger.warning("No API key provided for LLM ranking")
            return None
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert HR recruiter. Provide detailed, accurate evaluations in JSON format."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.3,
                "max_tokens": 1000
            }
            
            response = requests.post(self.base_url, headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                logger.error(f"LLM API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error calling LLM API: {str(e)}")
            return None
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response into structured data."""
        try:
            # Try to extract JSON from the response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start != -1 and json_end != 0:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            else:
                # Fallback parsing
                return self._fallback_parse_response(response)
                
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {str(e)}")
            return self._fallback_parse_response(response)
    
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