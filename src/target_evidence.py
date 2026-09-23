"""Collect target-disease evidence for TYK2 in ankylosing spondylitis from Open Targets.

Writes CSVs to data/raw/opentargets/:
  as_associated_targets.csv   top targets for ankylosing spondylitis, with evidence-type scores
  tyk2_as_evidence.csv        individual TYK2 <-> AS evidence records
  tyk2_as_gwas_loci.csv       fine-mapped variants and locus-to-gene scores at TYK2 GWAS loci
  jak_disease_scores.csv      JAK family association scores across immune-mediated diseases
  tyk2_drugs.csv              drugs and clinical candidates with TYK2 as a target
"""

from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "raw" / "opentargets"

API_URL = "https://api.platform.opentargets.org/api/v4/graphql"

AS_ID = "MONDO_0005306"  # ankylosing spondylitis
JAK_FAMILY = {
    "TYK2": "ENSG00000105397",
    "JAK1": "ENSG00000162434",
    "JAK2": "ENSG00000096968",
    "JAK3": "ENSG00000105639",
}
# Related immune-mediated diseases for context (names resolved via search)
COMPARATOR_DISEASES = [
    "ankylosing spondylitis",
    "psoriatic arthritis",
    "psoriasis",
    "Crohn disease",
    "ulcerative colitis",
    "rheumatoid arthritis",
    "systemic lupus erythematosus",
]


def query(q: str, variables: dict | None = None) -> dict:
    r = requests.post(API_URL, json={"query": q, "variables": variables or {}}, timeout=120)
    r.raise_for_status()
    payload = r.json()
    if "errors" in payload:
        raise RuntimeError(payload["errors"][0]["message"])
    return payload["data"]


def disease_id(name: str) -> str:
    q = """query($q:String!){search(queryString:$q, entityNames:["disease"], page:{index:0,size:1})
           {hits{id name}}}"""
    return query(q, {"q": name})["search"]["hits"][0]["id"]


def as_associated_targets(n: int = 100) -> pd.DataFrame:
    q = """query($d:String!, $n:Int!){disease(efoId:$d){
             associatedTargets(page:{index:0,size:$n}){
               rows{target{approvedSymbol id} score datatypeScores{id score}}}}}"""
    rows = query(q, {"d": AS_ID, "n": n})["disease"]["associatedTargets"]["rows"]
    out = []
    for rank, r in enumerate(rows, 1):
        rec = {"rank": rank, "symbol": r["target"]["approvedSymbol"],
               "ensembl_id": r["target"]["id"], "overall_score": r["score"]}
        rec.update({d["id"]: d["score"] for d in r["datatypeScores"]})
        out.append(rec)
    return pd.DataFrame(out).fillna(0)


def tyk2_as_evidence() -> tuple[pd.DataFrame, pd.DataFrame]:
    q = """query($d:String!, $t:String!){disease(efoId:$d){
             evidences(ensemblIds:[$t], size:100){rows{
               datasourceId datatypeId score literature publicationYear clinicalStage
               drug{name id}
               credibleSet{studyLocusId
                 locus{rows{posteriorProbability variant{id rsIds mostSevereConsequence{label}}}}
                 l2GPredictions{rows{target{approvedSymbol} score}}}}}}}"""
    rows = query(q, {"d": AS_ID, "t": JAK_FAMILY["TYK2"]})["disease"]["evidences"]["rows"]

    evidence, loci = [], []
    for r in rows:
        evidence.append({
            "datasource": r["datasourceId"],
            "datatype": r["datatypeId"],
            "score": r["score"],
            "drug": (r.get("drug") or {}).get("name"),
            "clinical_stage": r.get("clinicalStage"),
            "publication_year": r.get("publicationYear"),
            "pmids": ";".join(r.get("literature") or []),
        })
        cs = r.get("credibleSet")
        if cs:
            l2g = {p["target"]["approvedSymbol"]: p["score"] for p in cs["l2GPredictions"]["rows"]}
            top_gene = max(l2g, key=l2g.get)
            for v in cs["locus"]["rows"]:
                loci.append({
                    "study_locus_id": cs["studyLocusId"],
                    "pmid": ";".join(r.get("literature") or []),
                    "rsid": ";".join(v["variant"]["rsIds"] or []),
                    "variant_id": v["variant"]["id"],
                    "consequence": (v["variant"]["mostSevereConsequence"] or {}).get("label"),
                    "posterior_probability": v["posteriorProbability"],
                    "l2g_tyk2": l2g.get("TYK2"),
                    "l2g_top_gene": top_gene,
                    "l2g_top_score": l2g[top_gene],
                })
    return pd.DataFrame(evidence), pd.DataFrame(loci)


def jak_disease_scores() -> pd.DataFrame:
    q = """query($d:String!, $t:[String!]!){disease(efoId:$d){name
             associatedTargets(Bs:$t, page:{index:0,size:10}){
               rows{target{approvedSymbol} score datatypeScores{id score}}}}}"""
    out = []
    for name in COMPARATOR_DISEASES:
        did = disease_id(name)
        d = query(q, {"d": did, "t": list(JAK_FAMILY.values())})["disease"]
        for r in d["associatedTargets"]["rows"]:
            rec = {"disease": d["name"], "disease_id": did,
                   "target": r["target"]["approvedSymbol"], "overall_score": r["score"]}
            rec.update({x["id"]: x["score"] for x in r["datatypeScores"]})
            out.append(rec)
    return pd.DataFrame(out).fillna(0)


def tyk2_drugs() -> pd.DataFrame:
    q = """query($t:String!){target(ensemblId:$t){drugAndClinicalCandidates{rows{
             maxClinicalStage drug{name id drugType} diseases{disease{name}}}}}}"""
    rows = query(q, {"t": JAK_FAMILY["TYK2"]})["target"]["drugAndClinicalCandidates"]["rows"]
    out = []
    for r in rows:
        diseases = sorted({d["disease"]["name"] for d in r["diseases"] if d.get("disease")})
        out.append({
            "drug": r["drug"]["name"],
            "chembl_id": r["drug"]["id"],
            "drug_type": r["drug"]["drugType"],
            "max_stage": r["maxClinicalStage"],
            "n_indications": len(diseases),
            "tested_in_as": "ankylosing spondylitis" in diseases,
            "indications": "; ".join(diseases),
        })
    return pd.DataFrame(out)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("AS-associated targets...")
    as_associated_targets().to_csv(OUT_DIR / "as_associated_targets.csv", index=False)

    print("TYK2 <-> AS evidence...")
    evidence, loci = tyk2_as_evidence()
    evidence.to_csv(OUT_DIR / "tyk2_as_evidence.csv", index=False)
    loci.to_csv(OUT_DIR / "tyk2_as_gwas_loci.csv", index=False)

    print("JAK family scores across diseases...")
    jak_disease_scores().to_csv(OUT_DIR / "jak_disease_scores.csv", index=False)

    print("TYK2 drugs and clinical candidates...")
    tyk2_drugs().to_csv(OUT_DIR / "tyk2_drugs.csv", index=False)

    print(f"Saved Open Targets tables to {OUT_DIR}")


if __name__ == "__main__":
    main()
