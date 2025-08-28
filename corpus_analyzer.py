#!/usr/bin/env python3
"""
Corpus Analysis Tool for Academic Papers
Processes multiple papers in parallel and extracts structured insights
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
import tempfile
import subprocess
from dataclasses import dataclass, asdict
import argparse

# Try to import PDF processing libraries
try:
    import PyPDF2
    HAS_PDF = True
except ImportError:
    try:
        import pdfplumber
        HAS_PDF = True
    except ImportError:
        HAS_PDF = False

@dataclass
class ProcessingResult:
    file_path: str
    success: bool
    analysis: dict | None = None
    error: str | None = None
    processing_time: float | None = None

class CorpusAnalyzer:
    def __init__(self, 
                 papers_dir: Path, 
                 output_dir: Path,
                 model: str = "anthropic/claude-sonnet-4-20250514",
                 max_workers: int = 4,
                 resume: bool = False):
        self.papers_dir = Path(papers_dir)
        self.output_dir = Path(output_dir)
        self.model = model
        self.max_workers = max_workers
        self.resume = resume
        
        # Create output directories
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.results_dir = self.output_dir / "individual_results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Load schema
        schema_path = Path("paper_analysis_schema.json")
        if schema_path.exists():
            with open(schema_path) as f:
                self.schema = json.load(f)
        else:
            self.schema = None
            
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / "processing.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from PDF file"""
        if not HAS_PDF:
            raise RuntimeError("No PDF processing library available. Install PyPDF2 or pdfplumber.")
        
        text = ""
        try:
            # Try pdfplumber first (better text extraction)
            if 'pdfplumber' in sys.modules:
                import pdfplumber
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            else:
                # Fallback to PyPDF2
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
        except Exception as e:
            self.logger.error(f"Failed to extract text from {pdf_path}: {e}")
            raise
            
        return text.strip()

    def read_paper(self, paper_path: Path) -> str:
        """Read paper content from various formats"""
        if paper_path.suffix.lower() == '.pdf':
            return self.extract_text_from_pdf(paper_path)
        elif paper_path.suffix.lower() in ['.txt', '.md']:
            with open(paper_path, encoding='utf-8') as f:
                return f.read()
        else:
            raise ValueError(f"Unsupported file format: {paper_path.suffix}")

    def create_analysis_prompt(self, paper_text: str) -> str:
        """Create the analysis prompt for gptme"""
        schema_text = json.dumps(self.schema, indent=2) if self.schema else "No schema provided"
        
        prompt = f"""Analyze this academic paper and extract structured information according to the JSON schema provided.

IMPORTANT: Return ONLY a valid JSON object that follows the schema structure. Do not include any explanation text before or after the JSON.

JSON Schema:
{schema_text}

Paper Text:
{paper_text[:8000]}...  # Truncate for length

Instructions:
1. Extract all relevant information from the paper
2. Fill in as many fields as possible from the schema
3. Use empty arrays [] for missing list fields
4. Use null for missing single fields
5. Be concise but accurate
6. Focus on concrete, factual information
7. Rate innovation_score, technical_complexity, and practical_impact on 1-5 scale

Return the analysis as valid JSON:"""
        
        return prompt

    async def analyze_single_paper(self, paper_path: Path) -> ProcessingResult:
        """Analyze a single paper using gptme"""
        import time
        start_time = time.time()
        
        result_file = self.results_dir / f"{paper_path.stem}.json"
        
        # Skip if result exists and resume is enabled
        if self.resume and result_file.exists():
            try:
                with open(result_file) as f:
                    analysis = json.load(f)
                self.logger.info(f"Resumed: {paper_path.name}")
                return ProcessingResult(
                    file_path=str(paper_path),
                    success=True,
                    analysis=analysis,
                    processing_time=0
                )
            except json.JSONDecodeError:
                self.logger.warning(f"Corrupted result file {result_file}, reprocessing...")

        try:
            # Read paper content
            self.logger.info(f"Processing: {paper_path.name}")
            paper_text = self.read_paper(paper_path)
            
            if len(paper_text.strip()) < 100:
                raise ValueError("Paper text too short or empty")

            # Create analysis prompt
            prompt = self.create_analysis_prompt(paper_text)
            
            # Run gptme analysis
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(prompt)
                temp_prompt_file = f.name

            try:
                cmd = [
                    "gptme", 
                    "--model", self.model,
                    "--non-interactive",
                    "--system", "short",
                    f"cat {temp_prompt_file}"
                ]
                
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    timeout=300  # 5 minute timeout
                )
                
                if result.returncode != 0:
                    raise RuntimeError(f"gptme failed: {result.stderr}")
                    
                # Parse JSON from output
                output = result.stdout.strip()
                
                # Try to extract JSON from the output
                try:
                    # Look for JSON in the output
                    import re
                    json_match = re.search(r'\{.*\}', output, re.DOTALL)
                    if json_match:
                        json_str = json_match.group()
                        analysis = json.loads(json_str)
                    else:
                        # Try parsing the whole output as JSON
                        analysis = json.loads(output)
                except json.JSONDecodeError:
                    # If JSON parsing fails, create a minimal structure
                    analysis = {
                        "metadata": {"title": paper_path.stem},
                        "content_analysis": {"main_topic": "Parse failed"},
                        "keywords": ["parsing_error"],
                        "raw_output": output
                    }

                # Save individual result
                with open(result_file, 'w') as f:
                    json.dump(analysis, f, indent=2)

                processing_time = time.time() - start_time
                self.logger.info(f"Completed: {paper_path.name} ({processing_time:.1f}s)")
                
                return ProcessingResult(
                    file_path=str(paper_path),
                    success=True,
                    analysis=analysis,
                    processing_time=processing_time
                )

            finally:
                Path(temp_prompt_file).unlink(missing_ok=True)

        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Failed: {paper_path.name} - {str(e)}")
            
            # Save error result
            error_analysis = {
                "metadata": {"title": paper_path.stem},
                "content_analysis": {"main_topic": "Processing failed"},
                "keywords": ["error"],
                "processing_error": str(e)
            }
            
            with open(result_file, 'w') as f:
                json.dump(error_analysis, f, indent=2)
            
            return ProcessingResult(
                file_path=str(paper_path),
                success=False,
                error=str(e),
                processing_time=processing_time
            )

    async def process_all_papers(self) -> list[ProcessingResult]:
        """Process all papers in the corpus"""
        # Find all paper files
        paper_files = []
        for ext in ['*.pdf', '*.txt', '*.md']:
            paper_files.extend(self.papers_dir.glob(ext))
            paper_files.extend(self.papers_dir.rglob(ext))  # Recursive search
        
        paper_files = list(set(paper_files))  # Remove duplicates
        self.logger.info(f"Found {len(paper_files)} papers to process")
        
        if not paper_files:
            self.logger.warning("No papers found!")
            return []

        # Process papers with limited concurrency
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def process_with_semaphore(paper_path):
            async with semaphore:
                return await self.analyze_single_paper(paper_path)

        # Process all papers
        tasks = [process_with_semaphore(paper) for paper in paper_files]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Task failed for {paper_files[i]}: {result}")
                processed_results.append(ProcessingResult(
                    file_path=str(paper_files[i]),
                    success=False,
                    error=str(result)
                ))
            else:
                processed_results.append(result)

        return processed_results

    def generate_summary_report(self, results: list[ProcessingResult]):
        """Generate summary statistics and aggregated results"""
        successful_results = [r for r in results if r.success and r.analysis]
        
        summary = {
            "processing_summary": {
                "total_papers": len(results),
                "successful": len(successful_results),
                "failed": len(results) - len(successful_results),
                "success_rate": len(successful_results) / len(results) if results else 0
            },
            "corpus_statistics": self._generate_corpus_stats(successful_results),
            "all_results": [asdict(r) for r in results]
        }
        
        # Save summary
        with open(self.output_dir / "corpus_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Generate human-readable report
        self._generate_readable_report(summary)
        
        return summary

    def _generate_corpus_stats(self, results: list[ProcessingResult]) -> dict:
        """Generate statistics from successful analyses"""
        if not results:
            return {}
        
        stats = {
            "total_analyzed": len(results),
            "domains": {},
            "techniques": {},
            "innovations": {},
            "research_types": {}
        }
        
        for result in results:
            analysis = result.analysis
            if not analysis:
                continue
                
            # Count domains
            domain = analysis.get("content_analysis", {}).get("research_domain", "Unknown")
            stats["domains"][domain] = stats["domains"].get(domain, 0) + 1
            
            # Count techniques
            techniques = analysis.get("methodology", {}).get("techniques_used", [])
            for tech in techniques:
                stats["techniques"][tech] = stats["techniques"].get(tech, 0) + 1
            
            # Count research types
            research_type = analysis.get("methodology", {}).get("research_type", "Unknown")
            stats["research_types"][research_type] = stats["research_types"].get(research_type, 0) + 1
            
            # Count innovations
            innovations = analysis.get("innovations", {}).get("innovations_claimed", [])
            for innovation in innovations:
                stats["innovations"][innovation] = stats["innovations"].get(innovation, 0) + 1
        
        # Sort by frequency
        for category in stats:
            if isinstance(stats[category], dict):
                stats[category] = dict(sorted(stats[category].items(), key=lambda x: x[1], reverse=True))
        
        return stats

    def _generate_readable_report(self, summary: dict):
        """Generate a human-readable markdown report"""
        stats = summary.get("corpus_statistics", {})
        proc_summary = summary.get("processing_summary", {})
        
        report = f"""# Corpus Analysis Report

## Processing Summary
- **Total Papers**: {proc_summary.get('total_papers', 0)}
- **Successfully Processed**: {proc_summary.get('successful', 0)}
- **Failed**: {proc_summary.get('failed', 0)}
- **Success Rate**: {proc_summary.get('success_rate', 0)*100:.1f}%

## Research Domains
"""
        
        domains = stats.get("domains", {})
        for domain, count in list(domains.items())[:10]:  # Top 10
            report += f"- **{domain}**: {count} papers\n"
        
        report += "\n## Most Common Techniques\n"
        techniques = stats.get("techniques", {})
        for tech, count in list(techniques.items())[:15]:  # Top 15
            report += f"- **{tech}**: {count} papers\n"
        
        report += "\n## Research Types\n"
        research_types = stats.get("research_types", {})
        for rtype, count in research_types.items():
            report += f"- **{rtype}**: {count} papers\n"
        
        # Save report
        with open(self.output_dir / "corpus_report.md", 'w') as f:
            f.write(report)
        
        print(f"\n{report}")

async def main():
    parser = argparse.ArgumentParser(description="Analyze academic paper corpus")
    parser.add_argument("papers_dir", help="Directory containing papers to analyze")
    parser.add_argument("-o", "--output", default="corpus_analysis_output", 
                       help="Output directory for results")
    parser.add_argument("-m", "--model", default="anthropic/claude-sonnet-4-20250514",
                       help="LLM model to use")
    parser.add_argument("-w", "--workers", type=int, default=4,
                       help="Number of parallel workers")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from existing results")
    
    args = parser.parse_args()
    
    analyzer = CorpusAnalyzer(
        papers_dir=args.papers_dir,
        output_dir=Path(args.output),
        model=args.model,
        max_workers=args.workers,
        resume=args.resume
    )
    
    print("Starting corpus analysis...")
    print(f"Papers directory: {args.papers_dir}")
    print(f"Output directory: {args.output}")
    print(f"Model: {args.model}")
    print(f"Workers: {args.workers}")
    
    results = await analyzer.process_all_papers()
    summary = analyzer.generate_summary_report(results)
    
    print("\nAnalysis complete!")
    print(f"Results saved to: {args.output}")
    print(f"Success rate: {summary['processing_summary']['success_rate']*100:.1f}%")

if __name__ == "__main__":
    asyncio.run(main())
