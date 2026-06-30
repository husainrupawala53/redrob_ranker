import  pandas as pd 
import numpy  as np
from datetime import date
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import argparse as argparse

JD_text="""senior ai engineer  machine learning embeddings retrieval ranking vector database 
 production deployment  evaluation ndcg mrr map hybrid search pytorch tensorflow  
llm fine-tuning lora qlora recommendation system faiss  pinecone qdrant weaviate
python backend inference optimization senior software engineer product company startup """

w={
    "ai_skills":0.22,
    "assessments":0.09,
    "career":0.22,
    "ai_trajectory":0.04,
    "experience":0.13,
    "availability": 0.15,
    "salary":0.04,
    "edu_github":0.06,
    "market":0.05,
}
total=sum(w.values())
w={k:v/total for k,v in w.items()}
assert abs(sum(w.values())-1.0) <1e-9

def sigmoid(x,centre,scale):
    """Smooth 0->1 curve. High x -> low score (for 'lower is better' signals)."""
    return float(1/(1+np.exp((x-centre)/scale)))

def sigmoid_high(x,centre,scale):
    """Smooth 0->1 curve. High x->  low score (for 'lower is  better'signals)."""
    return float(1/(1+np.exp(-(x-centre)/scale)))

def fast_filter(df: pd.DataFrame) :
    """Remove obvious non-matches cheaply.
    NOT zero-scoring - just  hard disqaullifiers.
    Improvement J: reduces 100K -> ~10k before expensive scoring.
    """
    mask=(
        (df['years_of_experience']>=2)&
        (df['days_since_active']<= 180)&
        (df['recruiter_response_rate']>=0.05)&
        (df['open_to_work']!= False)
    )
    filtered=df[mask].copy()
    print(f'Stage 1 filter: {len(df):,}->{len(filtered)} canndidates')
    return filtered
def score_ai_skills(row):
    """
    AI skill count from CSV (pre-computed against AI_CORE  terms).
    Max expected = 8  AI core skills -> full score.
    Improvement B handeled sperately in score_assesments().
    """
    return min(row.get('ai_skill_count',0)/8,1.0)

def score_assessments(row):
    """
    Improvemnt B:  use verified Redrob platform assessment scores.
    avg_ai_assess_score=-1 means  no assessment  taken.
    """
    avg=row.get('avg_ai_assess_score',-1)
    if avg== -1:
        return 0.3 # no  assessments -neutral-low,not zero
    return avg/100.0

def score_career_tfidf(row,tfidf_scores:dict):
    """
    Semantic career relevance via TF-IDF cosine similarity.
    Replaces fragile title-only string matching.
    Also retains title checkas a multiplier to still catch  the keyword  trap .
    """
    cid=row['candidate_id']
    tfidf_sim=tfidf_scores.get(cid,0.0)

    title=str(row.get('current_title','')).lower()
    WRONG=['marketing','hr manager','accountant','sales executive',
         'graphic designer','customer support','content writer', 
         'civil engineer','mechanical  engineer','operations manager' ]
    if any(w in title for w in WRONG):
        title_mult=0.25
    elif any(t in title for t in ['ai','ml','macine learning','data scientist',
                                  'nlp','backend','software','research']):
        title_mult=1.0
    else:
        title_mult=0.7
    return tfidf_sim*title_mult
def score_ai_trajectory(row) :
    """
     how many months has this candidate spent in AI-relevant roles?
    Catches the semantic gap — someone who 'built recommendation systems'
    will have AI-relevant titles in career_history even if their skill list is sparse.
    """
    ai_months = row.get('ai_months_total', 0)
    # 24 months (2 years) of AI roles = full score
    return min(ai_months / 24, 1.0)

def score_experience(row) -> float:
    """
      smooth sigmoid instead of step function.
    Job Description  wants 5-9 years. Peak score at 7 years.
    """
    yoe = row.get('years_of_experience', 0)
    if 5 <= yoe <= 9:
        return 1.0
    if yoe < 5:
        # sigmoid rising from 0 to 1 as yoe approaches 5
        return sigmoid_high(yoe, centre=3.5, scale=0.8)
    # yoe > 9: sigmoid falling
    return sigmoid(yoe, centre=11, scale=1.5)
def score_availability(row) -> float:
    """
     soft curves + recency × RRR interaction.

    Key design decisions:
    - open_to_work=False → 0.5 multiplier, NOT 0.10 
      A passive but excellent candidate should still rank.
    - Recency and RRR are multiplied together, not averaged
      Active + responsive = great. Active + ghost = poor. Inactive + responsive = poor.
    """
    open_mult = 1.0 if row.get('open_to_work', False) else 0.5

    # Sigmoid recency: centre=45 days, scale=15
    days = row.get('days_since_active', 999)
    recency_s = sigmoid(days, centre=45, scale=15)

    # Sigmoid response rate: centre=0.3, scale=0.1
    rrr = row.get('recruiter_response_rate', 0)
    rrr_s = sigmoid_high(rrr, centre=0.3, scale=0.1)

    #  multiply recency × RRR (not average)
    engagement = recency_s * rrr_s

    # Interview completion rate — reliability signal
    icr = row.get('interview_completion_rate', 0)
    icr_s = sigmoid_high(icr, centre=0.5, scale=0.15)

    # Notice period: JD wants < 30 days
    notice = row.get('notice_period_days', 90)
    notice_s = sigmoid(notice, centre=45, scale=15)

    raw = 0.50 * engagement + 0.30 * icr_s + 0.20 * notice_s
    return raw * open_mult
def score_salary(row, budget_min=35, budget_max=55) :
    """
     check if candidate's salary expectations align with
    a Series A company budget (~35-55 LPA for a senior AI engineer).
    Does not disqualify — just adjusts score.
    """
    cmin = row.get('salary_min_lpa', 0)
    cmax = row.get('salary_max_lpa', 0)
    if cmin == 0 and cmax == 0:
        return 0.7   # unknown — neutral
    if cmin > budget_max * 1.3:
        return 0.2   # expects 30%+ above budget — serious mismatch
    if cmin > budget_max * 1.1:
        return 0.5   # expects 10-30% above — possible mismatch
    if cmax < budget_min * 0.7:
        return 0.7   # well under — possible under-qualification signal
    return 1.0       # within range

def score_edu_github(row) -> float:
    """
     soften the GitHub penalty.
    No GitHub (-1) now gets 0.40 (many senior engineers
    at large companies don't maintain public GitHub).
    Also adds edu_field_relevant as a modifier.
    """
    EDU = {'tier_1':1.0,'tier_2':0.80,'tier_3':0.60,'tier_4':0.40,'unknown':0.30}
    edu_s = EDU.get(row.get('best_edu_tier','unknown'), 0.30)
    if row.get('edu_field_relevant', False):
        edu_s = min(edu_s + 0.10, 1.0)  # bonus for CS/AI field of study

    gh = row.get('github_activity_score', -1)
    if gh == -1:
        gh_s = 0.40   # softer penalty 
    else:
        gh_s = sigmoid_high(gh, centre=40, scale=15)

    return 0.50 * edu_s + 0.50 * gh_s

def score_market_demand(row) -> float:
    """
    saved_by_recruiters_30d and search_appearance_30d
    are independent market demand signals. Combine with profile completeness.
    A candidate saved by 20 recruiters is more likely to be worth contacting
    than one saved by 0 — regardless of their profile content.
    """
    saved  = row.get('saved_by_recruiters_30d', 0)
    views  = row.get('profile_views_30d', 0)
    compl  = row.get('profile_completeness', 0) / 100

    saved_s = sigmoid_high(saved,  centre=5,  scale=3)
    views_s = sigmoid_high(views,  centre=20, scale=10)

    return 0.50 * saved_s + 0.30 * views_s + 0.20 * compl

def compute_tfidf_scores(df: pd.DataFrame) -> dict:
    """
    Improvement A: fit TF-IDF on career_text, compute cosine similarity to JD.
    Returns {candidate_id: similarity_score}.
    This runs in < 5 seconds on 10K candidates (CPU only, no network).
    """
    print('Computing TF-IDF career similarity...')
    texts  = df['career_text'].fillna('').tolist()
    all_texts = [JD_text] + texts
    vec    = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=2)
    matrix =csr_matrix (vec.fit_transform(all_texts))
    jd_vec = matrix[0]
    cand_vecs = matrix[1:]
    sims   = cosine_similarity(jd_vec, cand_vecs)[0]
    return dict(zip(df['candidate_id'].tolist(), sims.tolist()))


def compute_all_scores(df: pd.DataFrame) -> pd.DataFrame:
    tfidf_scores = compute_tfidf_scores(df)

    records = []
    for _, row in df.iterrows():
        sk  = score_ai_skills(row)
        ass = score_assessments(row)
        car = score_career_tfidf(row, tfidf_scores)
        trj = score_ai_trajectory(row)
        exp = score_experience(row)
        av  = score_availability(row)
        sal = score_salary(row)
        eg  = score_edu_github(row)
        mkt = score_market_demand(row)

        final = (sk  * w['ai_skills']   +
                 ass * w['assessments']  +
                 car * w['career']       +
                 trj * w['ai_trajectory']+
                 exp * w['experience']   +
                 av  * w['availability'] +
                 sal * w['salary']       +
                 eg  * w['edu_github']   +
                 mkt * w['market'])

        records.append({
            'candidate_id': row['candidate_id'],
            'final_score':  round(final, 4),
            's_skills':     round(sk, 3),
            's_assess':     round(ass, 3),
            's_career':     round(car, 3),
            's_trajectory': round(trj, 3),
            's_experience': round(exp, 3),
            's_availability':round(av, 3),
            's_salary':     round(sal, 3),
            's_edu_github': round(eg, 3),
            's_market':     round(mkt, 3),
        })

    return pd.DataFrame(records)

def build_reasoning(row, scores_row) -> str:
    """
    Improvement K: reasoning should be actionable for a recruiter/judge,
    not just a string of fields joined by semicolons.
    """
    parts = []
    parts.append(f"{row['current_title']} ({row['years_of_experience']:.1f} yrs exp)")

    ai_cnt = int(row.get('ai_skill_count', 0))
    parts.append(f"{ai_cnt} AI-core skills")

    avg_a = row.get('avg_ai_assess_score', -1)
    if avg_a > 0:
        parts.append(f"avg assessed {avg_a:.0f}/100 on AI skills")

    ai_mo = int(row.get('ai_months_total', 0))
    if ai_mo > 0:
        parts.append(f"{ai_mo // 12}y {ai_mo % 12}m in AI roles")

    days = int(row.get('days_since_active', 999))
    rrr  = row.get('recruiter_response_rate', 0)
    parts.append(f"active {days}d ago; RRR {rrr:.0%}")

    gh = row.get('github_activity_score', -1)
    if gh > 0:
        parts.append(f"GitHub {gh:.0f}")

    tier = row.get('best_edu_tier', 'unknown')
    if tier in ('tier_1', 'tier_2'):
        parts.append(tier.replace('_', '-') + ' institution')

    sal_min = row.get('salary_min_lpa', 0)
    sal_max = row.get('salary_max_lpa', 0)
    if sal_min > 0:
        parts.append(f"expects {sal_min:.0f}-{sal_max:.0f} LPA")

    return '; '.join(parts)

def rank(flat_csv='outputs/candidates_flat.csv',
         out_csv='outputs/submission.csv',
         top_n=100):

    print('Loading candidates...')
    df = pd.read_csv(flat_csv)
    df['last_active_date'] = pd.to_datetime(df['last_active_date'], errors='coerce')
    today = pd.Timestamp(date.today())
    df['days_since_active'] = (today - df['last_active_date']).dt.days.fillna(999)
    print(f'Loaded {len(df):,} candidates')

    # Stage 1: fast filter
    df = fast_filter(df)

    # Stage 2: full scoring
    scores_df = compute_all_scores(df)
    df = df.merge(scores_df, on='candidate_id')

    # Sort: score desc, then candidate_id asc (tie-break per submission rules)
    df = df.sort_values(['final_score', 'candidate_id'], ascending=[False, True])

    # FIX: don't ask for more rows than exist
    actual_n = min(top_n, len(df))
    if actual_n < top_n:
        print(f'⚠️  Only {actual_n} candidates survived filtering, '
              f'requested top {top_n}. Returning all {actual_n}.')

    top = df.head(actual_n).copy()
    top['rank'] = range(1, actual_n + 1)
    top['reasoning'] = top.apply(lambda r: build_reasoning(r, r), axis=1)

    sub = top[['candidate_id', 'rank', 'final_score', 'reasoning']].copy()
    sub.columns = ['candidate_id', 'rank', 'score', 'reasoning']

    Path('outputs').mkdir(exist_ok=True)
    sub.to_csv(out_csv, index=False)
    print(f'\nTop 10:')
    print(sub.head(10).to_string())
    print(f'\nSubmission written to {out_csv}')
    return sub


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--flat', default='outputs/candidates_flat.csv')
    ap.add_argument('--out',  default='outputs/submission.csv')
    a = ap.parse_args()
    rank(a.flat, a.out)









