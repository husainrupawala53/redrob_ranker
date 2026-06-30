import pandas as pd
def apply_company_cap(df,max_per_company=5):
    company_counts={}
    keep=[]
    for _,row in df.itterrows():
        company=str(row.get('current_company','unknown')).strip()
        count=company_counts.get(company,0)
        if count < max_per_company:
            keep.append(row)
            company_counts[company]=count+1
        if  len(keep)>=100:
            break
    result=pd.DataFrame(keep)
    result['rank']=range(1,len(result)+1)
    return result

