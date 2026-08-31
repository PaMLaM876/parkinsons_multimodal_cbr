import json
import os

log_path = r'C:\Users\Admin\.gemini\antigravity-ide\brain\f4e91fa7-46a4-46da-bb82-b4c011c3e2fb\.system_generated\logs\transcript_full.jsonl'
out_dir = r'C:\Users\Admin\Desktop\parkinsons_multimodal_cbr\showcase\recovered'
os.makedirs(out_dir, exist_ok=True)

count = 0
seen = set()

with open(log_path, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            data = json.loads(line)
            
            # Check tool calls
            if data.get('type') == 'PLANNER_RESPONSE':
                for tc in data.get('tool_calls', []):
                    args = tc.get('arguments', {})
                    if 'CodeContent' in args:
                        content = args['CodeContent']
                        if '<!DOCTYPE html>' in content:
                            if content not in seen:
                                seen.add(content)
                                with open(os.path.join(out_dir, f'recovered_{count}.html'), 'w', encoding='utf-8') as out_f:
                                    out_f.write(content)
                                count += 1
                                
            # Check tool responses
            elif data.get('type') == 'TOOL_RESPONSE':
                content = data.get('content', '')
                if '<!DOCTYPE html>' in content:
                    # Very basic extraction if it's embedded
                    if content not in seen:
                        seen.add(content)
                        with open(os.path.join(out_dir, f'recovered_tool_{count}.html'), 'w', encoding='utf-8') as out_f:
                            out_f.write(content)
                        count += 1
                        
        except Exception as e:
            pass

print(f"Recovered {count} HTML files!")
