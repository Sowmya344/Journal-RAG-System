import requests
import json
import pandas as pd
import re
import time
from typing import List, Dict, Optional
import random
from dataclasses import dataclass

@dataclass
class ProcessingConfig:
    """Configuration settings for text processing"""
    api_key: str = "2d5341da842ba4f133068a76fba1bc069199737b6178293b22765df2df33465f"
    model_name: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
    chunk_size: int = 1000
    overlap: int = 200
    max_tokens: int = 1000
    temperature: float = 0.3
    retries: int = 5
    base_backoff: int = 10
    rate_limit_delay: int = 12

class TextProcessor:
    """Main class for processing academic texts using LLM API"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.api_url = "https://api.together.xyz/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }

    def _clean_response(self, response: str) -> str:
        """Remove common LLM introductory phrases and clean response"""
        # Remove common intro phrases
        intro_patterns = [
            r"^Here's?\s+(a\s+)?(the\s+)?.*?summary.*?[:\-]\s*",
            r"^Here's?\s+(a\s+)?(the\s+)?.*?abstract.*?[:\-]\s*",
            r"^Here's?\s+(a\s+)?(the\s+)?.*?introduction.*?[:\-]\s*",
            r"^Here's?\s+(a\s+)?(the\s+)?.*?extract.*?[:\-]\s*",
            r"^Here\s+are\s+the.*?[:\-]\s*",
            r"^Based\s+on\s+.*?[:\-]\s*",
            r"^The\s+following\s+.*?[:\-]\s*",
            r"^Below\s+.*?[:\-]\s*",
            r"^\**\s*Summary\s*:\**\s*",
            r"^\**\s*Abstract\s*:\**\s*",
            r"^\**\s*Introduction\s*:\**\s*",
            r"^\**\s*Keywords?\s*:\**\s*"
        ]
        
        cleaned = response.strip()
        for pattern in intro_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
        
        # Clean up extra whitespace and newlines
        cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)
        cleaned = re.sub(r'^\s+', '', cleaned, flags=re.MULTILINE)
        
        return cleaned.strip()

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        for i in range(0, len(words), self.config.chunk_size - self.config.overlap):
            chunk = ' '.join(words[i:i + self.config.chunk_size])
            chunks.append(chunk)
            if i + self.config.chunk_size >= len(words):
                break
        return chunks

    def _call_llm(self, prompt: str, text_chunk: str) -> str:
        """Make API call to LLM with retry logic"""
        payload = {
            "model": self.config.model_name,
            "messages": [
                {
                    "role": "system", 
                    "content": "You are an expert medical researcher. Provide direct, concise responses without introductory phrases like 'Here is' or 'The following'. Start immediately with the requested content."
                },
                {
                    "role": "user", 
                    "content": f"{prompt}\n\nText:\n{text_chunk}"
                }
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature
        }

        for attempt in range(self.config.retries):
            try:
                response = requests.post(
                    self.api_url, 
                    headers=self.headers, 
                    json=payload, 
                    timeout=30
                )
                response.raise_for_status()
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                return self._clean_response(content)
                
            except requests.exceptions.RequestException as e:
                if attempt < self.config.retries - 1:
                    wait_time = self.config.base_backoff * (2 ** attempt) + random.uniform(1, 5)
                    print(f"   Retry {attempt + 1}/{self.config.retries} in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                else:
                    return f"API Error: Unable to process after {self.config.retries} attempts"
                    
            except (KeyError, Exception) as e:
                if attempt < self.config.retries - 1:
                    time.sleep(self.config.base_backoff)
                else:
                    return f"Processing Error: {str(e)}"

    def extract_keywords(self, text: str) -> str:
        """Extract relevant keywords and medical terms"""
        prompt = """Extract important keywords, medical terms, and key phrases. 
        Focus on: medical terminology, treatment names, study methodology, statistical terms, key findings.
        Return only comma-separated terms without explanations."""
        
        result = self._call_llm(prompt, text[:2000])
        
        # Extract the most relevant line if multiple lines returned
        lines = [line.strip() for line in result.split('\n') if line.strip()]
        for line in lines:
            if ',' in line and len(line.split(',')) >= 3:
                return line
        return result

    def extract_abstract(self, text: str) -> str:
        """Generate or extract abstract"""
        prompt = """Generate a concise abstract (150-250 words) covering:
        - Study objective and background
        - Methodology and design
        - Key findings and results
        - Main conclusions
        Write in standard academic abstract format."""
        
        return self._call_llm(prompt, text[:3000])

    def extract_summary(self, text: str) -> str:
        """Generate comprehensive summary"""
        prompt = """Create a comprehensive summary (250-350 words) including:
        - Research objective and rationale
        - Study design and methodology
        - Key results and statistical findings
        - Clinical implications and significance
        - Limitations and future directions"""
        
        chunks = self._chunk_text(text)
        summaries = []
        
        # Process first 3 chunks for summary
        for i, chunk in enumerate(chunks[:3]):
            if i > 0:
                time.sleep(self.config.rate_limit_delay)
            
            chunk_prompt = f"{prompt}\n\nThis is section {i+1} of the document:"
            summary_part = self._call_llm(chunk_prompt, chunk)
            
            if not summary_part.startswith(("API Error", "Processing Error")):
                summaries.append(summary_part)

        # Combine summaries if multiple parts
        if len(summaries) > 1:
            time.sleep(self.config.rate_limit_delay)
            combine_prompt = "Synthesize these sections into one cohesive summary (250-350 words):"
            combined_text = "\n\n".join(summaries)
            final_summary = self._call_llm(combine_prompt, combined_text)
            return final_summary if not final_summary.startswith(("API Error", "Processing Error")) else summaries[0]
        elif summaries:
            return summaries[0]
        else:
            return "Error: Unable to generate summary due to processing issues"

    def extract_introduction(self, text: str) -> str:
        """Extract or generate introduction section"""
        prompt = """Extract or generate the introduction section including:
        - Background and context
        - Problem statement and gap in knowledge
        - Study rationale and significance
        - Research objective and aims
        Use standard academic introduction format."""
        
        return self._call_llm(prompt, text[:2500])

class ResultsManager:
    """Handle saving and displaying results"""
    
    @staticmethod
    def save_to_csv(results: Dict[str, str], filename: str = "academic_analysis.csv") -> None:
        """Save results to CSV file"""
        df = pd.DataFrame([results])
        df.to_csv(filename, index=False, encoding='utf-8')

    @staticmethod
    def display_results(results: Dict[str, str]) -> None:
        """Display results in organized format"""
        print("\n" + "="*80)
        print("ACADEMIC TEXT ANALYSIS RESULTS")
        print("="*80)
        
        for section, content in results.items():
            if section == 'Title':
                continue
            print(f"\n{section.upper()}:")
            print("-" * (len(section) + 1))
            print(content)
            print()

def main():
    """Main execution function"""
    # Academic text to process
    TEXT_TO_PROCESS = """
    original article

    characteristics, treatment pattern survival patient high-risk early hormone receptor-positive breast cancer french real-world settings: exploratory study canto cohort5

    giugliano1,2,3,4, bertaut5, blanc5, a.-l. martin6, gaudin6, fournier7, kieffer8, sauterey9, levy10,

    campone11, tarpin12, lerebours13, m.-a. mouret-reynier14, curigliano3,4, andré1,2, pistilli1,15

    rassy1,16

    1department medical oncology, gustave roussy, villejuif; 2inserm u981, gustave roussy, villejuif, france; 3department oncology hematology-oncology,

    university milano, milan; 4division new drug early drug development innovative therapies, european institute oncology, irccs, milan, italy; 5unité méthodologie, biostatistiques data-management, centre georges-françois leclerc, dijon; 6direction data partenariats, unicancer, paris; 7institut bergonié,

    bordeaux; 8institut cancérologie lorrainedalexis vautrin, vandoeuvre le nancy; 9institut cancérologie l'ouest, angers; 10centre françois baclesse, caen; 11institut cancérologie l'ouest, nantes saint herblain; 12institut paoli calmettes, marseille; 13institut curiedrené huguenin, saint cloud; 14centre jean perrin,

    clermont ferrand; 15inserm u1279, gustave roussy, villejuif; 16cesp, inserm u1018, université paris-saclay, villejuif, france

    available online november

    background: patient hormone receptor-positive, human epidermal growth factor receptor (her2)-negative breast cancer (hrþ bc) unfavorable feature increased risk relapse currently candidate additional treatment strategies. evaluated real-world clinicopathological characteristics, treatment pattern survival outcome patient within cancer toxicity study (canto, nct01993498)

    patient methods: retrospective analysis prospective data collected within canto 2022. patient high-risk hrþ deﬁned either identiﬁcation least four positive axillary lymph node (lns) one three positive axillary lns tumor size histologic grade (cohort

    deﬁnition 1-3 positive lns ki-67 20% also considered (cohort kaplanemeier method used survival analysis.

    results: patient high-risk hrþ represented 15.0%-19.6% hrþ (cohort respectively)

    canto cohort. patient cohort patient (49.0% lns, (26.0% tumor (57.6% grade iii tumors. 79.9% favorable charlson comorbidity score 88.1% stage ii/iiia.

    patient 10 lns accounted 11.8% (neo)adjuvant chemotherapy administered 94.2% endocrine therapy prescribed 97.3% mostly aromatase inhibitor discontinued 34.3% mainly adverse events. patient enrolled least year data extraction 5-year invasive disease-free survival year distant relapse-free survival 79.9% [95% conﬁdence interval (ci) 77.2% 82.4% 83.5% (95% 80.9% 85.7% respectively.

    conclusions: real-world study conﬁrms patient hrþ unfavorable clinicopathological feature risk relapse early adjuvant treatment trajectory, despite (neo)adjuvant chemotherapy. imperative implement innovative treatment approach high-risk patients, ideally adding early possible adjuvant treatment.

    key words: early breast cancer, high risk, abemaciclib, real-world data, adjuvant cdk4/6 inhibitor

    introduction

    breast cancer (bc) common malignancy among woman estimated 2.3 million new case reported annually remains among major cause cancer- related death around 0.7 million death year.1

    hormone receptor-positive human epidermal

    growth factor receptor (her2)-negative (hrþ bc) prevalent subtype, accounting 70% cases.2 majority patient hrþ diagnosed *correspondence to: elie rassy, department medical oncology, gustave

    roussy, rue edouard vaillant, villejuif 94805, france. tel: þ33-1-42-11- 42-11

    (e. rassy)

    @elierassy, @fedgiugliano

    5note: result reported manuscript previously partly pre- sented european society medical oncology breast meeting 2024; may 15-17; berlin, germany.

    2059-7029/ author(s) published elsevier ltd behalf

    european society medical oncology. open access article

    by-nc-nd license

    early stage, allowing curative intent. stan- dard therapeutic approach start surgery, followed endocrine therapy (et) often radiotherapy chemotherapy. early breast cancer trialists' collabora- tive group reported higher risk bc-related event patient large primary tumor size, high lymph node (ln) involvement, high histologic grade ki- 20%.3 according international guidelines, patient feature might beneﬁt chemotherapy fol- lowed extended duration et, w20% experience disease relapse ﬁrst years.4-10 monarche trial established new standard care patient hrþ

    high risk relapse according deﬁnition take account anatomical stage (tumor size nodal involvement) biological intricacy reﬂected his- tologic grade.11-13 trial, addition cyclin- dependent kinase (cdk) 4/6 inhibitor abemaciclib year standard care reduced risk invasive disease-free survival (idfs) 4-year absolute beneﬁt 6.4% (85.8% versus 79.4% leading approval abemaciclib food drug administration eu- ropean medicine agency.11,13,14 goal,

    natalee trial shown addition ribociclib year standard care improved 3-year idf [hazard ratio 0.75, 95% conﬁdence interval (ci) 0.62-0.91] 3.0% 3.2% respectively, patient stage stage iii disease. however, longer follow-up needed better establish future position drug treatment landscape early bc.15,16

    randomized controlled trials, monarche

    natalee, represent benchmark evidence-based medicine, offering rigorous methodology minimizes bias confounding variables. however, stringent eligibility criterion might exclude patient poorer health performance

    status

    multiple

    comorbidities,

    thus

    limiting generalizability ﬁndings broader population.17-22 limitation addressed analysis real-world data (rwd) provide promising tool improve understanding clinical demographic feature patient could beneﬁt escalation adjuvant treatment. present study, aim describe clinicopathological characteristic survival outcome patient hrþ

    high risk relapse, using rwd reported

    cancer toxicity (canto) prospective study.23

    patient method

    data source

    obtained data canto study (nct01993498) prospective observational study collecting detailed tumor, treatment, toxicities, health-related patient-reported out- come biological data relevant since march (enrolment still ongoing).23 canto enrolling patient 18 year age, primary diag- nosis invasive stage ct0-ct3, cn0-3 previous treatment bc. patient assessed diagnosis shortly primary treatment (i.e. primary surgery, chemotherapy radiotherapy, whichever came last) time prescription, indicated, year initial post-primary treatment evaluation.
    """

    # Validate input
    if "REPLACE THIS" in TEXT_TO_PROCESS or not TEXT_TO_PROCESS.strip():
        print("❌ Error: Please provide valid text to process")
        return

    # Initialize processor
    config = ProcessingConfig()
    processor = TextProcessor(config)
    
    # Clean input text
    text = re.sub(r'\s+', ' ', TEXT_TO_PROCESS).strip()
    word_count = len(text.split())
    
    print(f"🚀 Processing {word_count:,} words...")
    print("="*60)

    # Process text sections
    sections = [
        ("Keywords", processor.extract_keywords),
        ("Abstract", processor.extract_abstract),
        ("Summary", processor.extract_summary),
        ("Introduction", processor.extract_introduction)
    ]

    results = {
        "Title": "Characteristics, treatment patterns and survival of patients with high-risk early hormone receptor-positive breast cancer in French real-world settings"
    }

    for i, (section_name, extract_func) in enumerate(sections, 1):
        print(f"[{i}/{len(sections)}] Extracting {section_name.lower()}...")
        
        try:
            result = extract_func(text)
            results[section_name] = result
            
            if i < len(sections):  # Don't wait after the last section
                time.sleep(config.rate_limit_delay)
                
        except Exception as e:
            print(f"   ⚠️  Error processing {section_name}: {e}")
            results[section_name] = f"Error: Could not process {section_name.lower()}"

    # Save and display results
    output_file = "academic_analysis_results.csv"
    ResultsManager.save_to_csv(results, output_file)
    ResultsManager.display_results(results)
    
    print(f"💾 Results saved to '{output_file}'")
    print("✅ Processing complete!")

if __name__ == "__main__":
    main()