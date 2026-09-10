"""Export a seeded, index-wide sample for two blinded human annotations."""
import argparse
import asyncio
import csv
import hashlib
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


async def sample_edges(service, namespace, size=100, seed=42):
    if not re.fullmatch(r"[A-Za-z0-9_]+",namespace):
        raise ValueError("Invalid graph namespace")
    label = f"PR_{namespace}_Chunk"
    rng = random.Random(seed)
    chosen, total, after_src, after_dst = [], 0, "", ""
    while True:
        rows = await service.execute_query(
            f"MATCH (s:{label})-[:HOP_ANSWER]->(t:{label}) "
            "WHERE s.id > $src OR (s.id=$src AND t.id > $dst) "
            "RETURN DISTINCT s.id AS src,t.id AS dst ORDER BY src,dst LIMIT 1000", {"src":after_src,"dst":after_dst})
        if not rows:
            break
        for row in rows:
            total += 1
            if len(chosen) < size:
                chosen.append(row)
            else:
                slot = rng.randrange(total)
                if slot < size:
                    chosen[slot] = row
        after_src, after_dst = rows[-1]["src"], rows[-1]["dst"]
    output = []
    for edge in sorted(chosen,key=lambda r:(r["src"],r["dst"])):
        rows = await service.execute_query(
            f"MATCH (s:{label} {{id:$src}})-[:HOP_ANSWER]->(t:{label} {{id:$dst}}) "
            "OPTIONAL MATCH (s)-[:HAS_Q_PLUS]->(q)-[:ANSWERED_BY]->(a)<-[:HAS_Q_MINUS]-(t) "
            "WITH s,t,q,a ORDER BY q.id,a.id "
            "RETURN s.title AS source_title,s.text AS source_text,t.title AS destination_title,t.text AS destination_text,"
            "collect(CASE WHEN q IS NULL THEN null ELSE {question:q.text,answerable_question:a.text} END) AS matched_question_pairs",edge)
        if len(rows) != 1:
            raise ValueError("Sampled edge disappeared or has ambiguous owners")
        sample_id = hashlib.sha256(f"{edge['src']}\0{edge['dst']}".encode()).hexdigest()[:20]
        output.append({"sample_id":sample_id,**rows[0]})
    return {"population_edges":total,"sample_size":len(output),"seed":seed,"items":output}


async def run(args):
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    from core.neo4j_service import Neo4jService
    service = Neo4jService()
    try:
        state = await service.execute_query("MATCH (m:RAGIndexSnapshot {strategy:'prehop',index_namespace:$namespace}) RETURN m.status AS status",{"namespace":args.namespace})
        if not state or any(row["status"] != "complete" for row in state):
            raise ValueError("Sample only a completed graph")
        result = await sample_edges(service,args.namespace,seed=args.seed)
        args.output.mkdir(parents=True,exist_ok=False)
        (args.output/'sample.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
        for person in (1,2):
            with (args.output/f'annotator_{person}.csv').open('x',newline='') as f:
                writer=csv.writer(f)
                writer.writerow(['sample_id','supplies_source_question','meaningful_transition','notes'])
                writer.writerows([item['sample_id'],'','',''] for item in result['items'])
        (args.output/'instructions.txt').write_text('Independently label yes, no, or unclear. For body links, use not_applicable for supplies_source_question. Inspect every matched question pair. Do not consult query retrieval scores. Reconcile disagreements only after saving both independent files. A and B use this same sample.\n')
    finally:
        await service.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--namespace',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    if args.execute:
        asyncio.run(run(args))
    else:
        print(json.dumps({'namespace':args.namespace,'output':str(args.output),'seed':args.seed,'edges':100}))


if __name__ == '__main__':
    main()
