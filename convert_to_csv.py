import json,csv,argparse
from pathlib import Path
from tqdm import tqdm
PROFIENCY_WEIGHT={'beginner':1,'Beginner':1,'intermediate':2,'Intermediate':2,'advanced':3,'Advanced':3,'expert':4,"Expert":4}

AI_CORE={
    "machine learning","deep learning",'nlp','llm','rag','embeddings','retrieval',
    'ranking','fine-tuning llms','transformers','pytorch','tensorflow','vector database','recommendation','faiss','pinecone','qdrant','weaviate','openai','hugging face','mlops','a/b testing','model deployment','evaluation','xgboost','scikit-learn','image classification','speec recognition','computer vision','generative AI'}

def skill_sort_key(s):
    pw=PROFIENCY_WEIGHT.get(s.get('proficiency','beginner'),1)
    return pw*(s.get('endorsements',0)+1)

def extract_flat_row(c):
    p=c.get('profile',{})
    rs=c.get('redrob_signals',{})
    edu=c.get('education',[])
    skills=c.get('skills',[])
    certs=c.get('certifications',[])
    career=c.get('career_history',[])
    #Education
    tier_order={'tier_1':4,'tier_2':3,'tier_3':2,'tier_4':1,'unknown':0}
    best_tier=max((tier_order.get(e.get('tier','unknown'),0)for e in edu),default=0)
    tier_label={4:'tier_1',3:'tier_2',2:'tier_3',1:'tier_4',0:'unknown'}[best_tier]
     
    cs_fields={'computer science','cs','information technology','ai','data science','macine learning','electronics','electrical','software engineering'}
    edu_field_relevant=any(any(f in e.get('field_of_study','').lower() for f in cs_fields)for e in edu)

    sorted_skills=sorted(skills,key=skill_sort_key,reverse=True)
    top5_skills='|'.join(s['name'] for s in sorted_skills[:5])
    ai_skill_count=sum(1 for  s in skills if s['name'].lower() in AI_CORE)
    expert_skills=[s['name'] for s in skills if  s.get('proficiency','').lower() in ('advanced','expert')]

    assessments = rs.get('skill_assessment_scores', {})

    ai_assess_scores = [
    v for k, v in assessments.items()
    if k.lower().strip() in AI_CORE
]

    best_ai_assess = (
    max(ai_assess_scores)
    if ai_assess_scores else -1
)

    avg_ai_assess = (
    sum(ai_assess_scores) / len(ai_assess_scores)
    if ai_assess_scores else -1
)

    ai_title_kw={'ai','ml','macine learning','data sciientist','recommendation','ranking'}
    ai_months_total=sum(
        j.get('duration_monts',0)for j in career
         if  any (kw in j.get('title','').lower()  for  kw in ai_title_kw) 
         )

    recent_roles=sorted(career,key=lambda j:j.get('start_date',''),reverse=True)[:3]
    career_text=''.join(
        j.get('title','')+''+j.get('description','') for j in recent_roles
    ).lower()

    sal=rs.get('expected_salary_range_inr_lpa',{})


    return {
        #identity
        'candidate_id':c['candidate_id'],
        'anonymized_name':p.get('anonymized_name',''),
        #Profile
        'headline':p.get('headline',''),
        'location':p.get('location',''),
        'country':p.get('country',''),
        'years_of_experience':p.get('years_of_experience',()),
        'current_title': p.get('current_title',''),
        'current_company':p.get('current_company',''),
        'current_company_size':p.get('current_company_size',''),
        'current_industry':p.get('current_industry',''),
        #Education
        'best_edu_tier': tier_label,
        'edu_field_relevant': edu_field_relevant,
        #Skills
        'total_skills':len(skills),
        'ai_skill_count':ai_skill_count,
        'expert_skill_count':len(expert_skills),
        'top5_skills':top5_skills,           
        # sorted by proficiency*endorsements
        'best_ai_assess_score':best_ai_assess,
        'avg_ai_assess_score':avg_ai_assess,
        #Career trajectory
        'ai_months_total':ai_months_total,
        'career_text':career_text[:2000],
        #Certifications
        'cert_count':len(certs),
        #Redrob signals
        'profile completeness':rs.get('profile_completeness_score',0),
        'signup_date':              rs.get('signup_date',''),
        'last_active_date':         rs.get('last_active_date',''),
        'open_to_work':             rs.get('open_to_work_flag',False),
        'profile_views_30d':        rs.get('profile_views_received_30d',0),
        'applications_30d':         rs.get('applications_submitted_30d',0),
        'recruiter_response_rate':  rs.get('recruiter_response_rate',0),
        'avg_response_time_hours':  rs.get('avg_response_time_hours',999),
        'connection_count':         rs.get('connection_count',0),
        'endorsements_received':    rs.get('endorsements_received',0),
        'notice_period_days':       rs.get('notice_period_days',90),
        'salary_min_lpa':           sal.get('min',0),
        'salary_max_lpa':           sal.get('max',0),
        'preferred_work_mode':      rs.get('preferred_work_mode',''),
        'willing_to_relocate':      rs.get('willing_to_relocate',False),
        'github_activity_score':    rs.get('github_activity_score',-1),
        'search_appearance_30d':    rs.get('search_appearance_30d',0),
        'saved_by_recruiters_30d':  rs.get('saved_by_recruiters_30d',0),
        'interview_completion_rate':rs.get('interview_completion_rate',0),
        'offer_acceptance_rate':    rs.get('offer_acceptance_rate',-1),
        'verified_email':           rs.get('verified_email',False),
        'verified_phone':           rs.get('verified_phone',False),
        'linkedin_connected':       rs.get('linkedin_connected',False),
    }
def extract_skill_rows(c):
    rs=c.get('redrob_signals',{})
    assessment=rs.get('skill_assessment_scores',{})
    return [{'candidate_id':c['candidate_id'],'skill_name':s['name'],
             'proficiency':s.get('proficiency',''),'endorsements':s.get('endorsements',0),
             'duration_months':s.get('duration_months',0),
             'assessment_score':assessment.get(s['name'],'')}
            for s in c.get('skills',[])]

def extract_career_rows(c):
    return [{'candidate_id':c['candidate_id'],
             'company':j.get('company',''),
             'title':j.get('title',''),
             'industry':j.get('industry',''),
             'company_size':j.get('company_size',''),
             'start_date':j.get('start_date',''),
             'end_date':j.get('end_date',''),
             'duration_months':j.get('duration_months',0),
             'is_current':j.get('is_current',False)}
            for j in c.get('career_history',[])]


def convert(input_path: str = 'data/candidates.jsonl', output_dir: str = 'outputs'):
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    flat_path   = output_dir / 'candidates_flat.csv'
    skills_path = output_dir / 'candidates_skills.csv'
    career_path = output_dir / 'candidates_career.csv'

    flat_fieldnames = [
        'candidate_id','anonymized_name','headline','location','country',
        'years_of_experience','current_title','current_company','current_company_size',
        'current_industry','best_edu_tier','edu_field_relevant','total_skills',
        'ai_skill_count','expert_skill_count','top5_skills','best_ai_assess_score',
        'avg_ai_assess_score','ai_months_total','career_text','cert_count',
        'profile completeness','signup_date','last_active_date','open_to_work',
        'profile_views_30d','applications_30d','recruiter_response_rate',
        'avg_response_time_hours','connection_count','endorsements_received',
        'notice_period_days','salary_min_lpa','salary_max_lpa','preferred_work_mode',
        'willing_to_relocate','github_activity_score','search_appearance_30d',
        'saved_by_recruiters_30d','interview_completion_rate','offer_acceptance_rate',
        'verified_email','verified_phone','linkedin_connected',
    ]
    skill_fieldnames = ['candidate_id','skill_name','proficiency','endorsements',
                        'duration_months','assessment_score']
    career_fieldnames = ['candidate_id','company','title','industry','company_size',
                         'start_date','end_date','duration_months','is_current']

    is_jsonl = str(input_path).endswith('.jsonl')

    with open(flat_path, 'w', newline='', encoding='utf-8') as ff, \
         open(skills_path, 'w', newline='', encoding='utf-8') as fs, \
         open(career_path, 'w', newline='', encoding='utf-8') as fc:

        fw = csv.DictWriter(ff, fieldnames=flat_fieldnames)
        sw = csv.DictWriter(fs, fieldnames=skill_fieldnames)
        cw = csv.DictWriter(fc, fieldnames=career_fieldnames)
        fw.writeheader(); sw.writeheader(); cw.writeheader()

        with open(input_path, 'r', encoding='utf-8') as f:
            if is_jsonl:
                candidates = (json.loads(line.strip()) for line in f if line.strip())
            else:
                candidates = iter(json.load(f))   # sample_candidates.json is a JSON array

            for cand in tqdm(candidates, desc='Processing candidates'):
                fw.writerow(extract_flat_row(cand))
                sw.writerows(extract_skill_rows(cand))
                cw.writerows(extract_career_rows(cand))

    print('Done')
    print(output_dir)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='data/candidates.jsonl',
                    help='Path to .jsonl or .json input file')
    ap.add_argument('--output_dir', default='outputs')
    args = ap.parse_args()
    convert(args.input, args.output_dir)