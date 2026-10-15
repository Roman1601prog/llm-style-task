"""Recover saved texts after Colab expiry; never invent lost telemetry."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def parse_review(text):
    sections=list(re.finditer(r'^## (type[12]) / (type[12]-\d+)\s*$',text,re.M))
    result={}
    for index,section in enumerate(sections):
        end=sections[index+1].start() if index+1<len(sections) else len(text)
        block=text[section.end():end]
        source=re.search(r'\A\s*Исходник:\s*\n(.*?)\n### baseline\s*\n',block,re.S)
        if not source: raise ValueError('Не найден исходник в REVIEW.md: '+section[2])
        for mode in ('baseline','prompt-only','adapted'):
            match=re.search(r'^### '+re.escape(mode)+r'\s*\n(.*?)\nПодсказки: ([^\n]*)',block,re.S|re.M)
            if not match: raise ValueError('Не найден ответ '+section[2]+'/'+mode)
            key=(section[1],section[2],mode)
            if key in result: raise ValueError('Повтор в REVIEW.md')
            warnings=ast.literal_eval(match[2])
            if not isinstance(warnings,list) or not all(isinstance(x,str) for x in warnings):
                raise ValueError('Неверный список подсказок')
            result[key]={'source':source[1].strip(),'raw_text':match[1].strip(),'warnings':warnings}
    if len(result)!=30: raise ValueError('В REVIEW.md должно быть 30 ответов')
    return result


def parse_posts(text):
    chunks=re.split(r'^===== id=(\d+) =====\s*$',text,flags=re.M)
    ids=[int(x) for x in chunks[1::2]]
    bodies=[x.strip() for x in chunks[2::2]]
    if chunks[0].strip() or ids!=[1,2,3,4,5] or len(bodies)!=5 or not all(bodies):
        raise ValueError('В TXT нужны пять постов id=1..5')
    return dict(zip(ids,bodies))


def keep_or_write(path,data):
    if path.exists():
        if path.read_bytes()!=data:
            raise ValueError('Уже существует другая версия '+str(path)+'. Файл не перезаписан.')
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(data)


def jsonl_bytes(rows):
    return ''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows).encode('utf-8')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='/content/llm-style-task')
    p.add_argument('--inputs',type=Path,default=Path(__file__).parent/'saved_files')
    args=p.parse_args();root=Path(args.root).resolve();inputs=args.inputs.resolve()
    manifest=json.loads((inputs/'SNAPSHOT_MANIFEST.json').read_text(encoding='utf-8'))
    for name,digest in manifest['saved_files_sha256'].items():
        if sha((inputs/name).read_bytes())!=digest: raise ValueError('Изменён файл комплекта: '+name)
    if sha((root/'data/raw/test_inputs.jsonl').read_bytes())!=manifest['test_inputs_sha256']:
        raise ValueError('Другие test-исходники в восстановленном проекте')
    for style in ('type1','type2'):
        for name in ('adapter_model.safetensors','adapter_config.json'):
            if not (root/f'adapters/{style}/{name}').is_file(): raise ValueError('Не восстановлен адаптер '+style)
    sys.path.insert(0,str(root))
    common=importlib.import_module('common')
    reviewed=parse_review((inputs/'REVIEW.md').read_text(encoding='utf-8'))
    # The checkpoint keeps these twenty original logs; verify the saved review is the same run.
    for mode in ('baseline','prompt-only'):
        existing=read_jsonl(root/f'artifacts/{mode}_validation.jsonl')
        keys={(r['style'],str(r['id']),mode) for r in existing}
        expected={k for k in reviewed if k[2]==mode}
        if len(existing)!=10 or keys!=expected: raise ValueError('Несовместимый checkpoint: '+mode)
        for r in existing:
            saved=reviewed[(r['style'],str(r['id']),mode)]
            if r['source'].strip()!=saved['source'] or r['raw_text'].strip()!=saved['raw_text'] or r['warnings']!=saved['warnings']:
                raise ValueError('REVIEW.md не соответствует checkpoint: '+mode+'/'+str(r['id']))
    rows=[]
    for style in ('type1','type2'):
        for original in read_jsonl(root/f'data/processed/{style}_val.jsonl'):
            saved=reviewed[(style,str(original['id']),'adapted')]
            if original['input'].strip()!=saved['source']: raise ValueError('Другой validation-исходник')
            rows.append({'id':original['id'],'style':style,'mode':'adapted','split':'validation',
                'source':saved['source'],'reference':original.get('target'),'raw_text':saved['raw_text'],
                'text':common.render_post(style,saved['raw_text']),'warnings':saved['warnings'],
                'prompt_tokens':None,'generated_tokens':None,'generation_seconds':None,'finish_reason':None,
                'recovery':{'from':'REVIEW.md','sha256':manifest['saved_files_sha256']['REVIEW.md'],
                    'raw_text':'saved verbatim in review','text':'rendered from saved raw_text using existing common.render_post',
                    'missing_original_telemetry':True}})
    validation=rows
    sources={r['id']:r['text'] for r in read_jsonl(root/'data/raw/test_inputs.jsonl')}
    final=[]
    for style in ('type1','type2'):
        name=f'outputs_{style}.txt'
        for item_id,post in parse_posts((inputs/name).read_text(encoding='utf-8')).items():
            final.append({'id':item_id,'style':style,'mode':'adapted','split':'test','source':sources[item_id],
                'text':post,'raw_text':None,'warnings':None,'prompt_tokens':None,'generated_tokens':None,
                'generation_seconds':None,'finish_reason':None,
                'recovery':{'from':name,'sha256':manifest['saved_files_sha256'][name],
                    'text':'saved final TXT post, unchanged','missing_original_raw_text':True,
                    'missing_original_telemetry':True}})
    # Validate all reconstructed content before writing any result.
    for name in manifest['saved_files_sha256']:
        keep_or_write(root/name,(inputs/name).read_bytes())
    keep_or_write(root/'artifacts/adapted_validation.jsonl',jsonl_bytes(validation))
    keep_or_write(root/'artifacts/adapted_test.jsonl',jsonl_bytes(final))
    recovery=root/'recovery';recovery.mkdir(exist_ok=True)
    keep_or_write(recovery/'SNAPSHOT_MANIFEST.json',(inputs/'SNAPSHOT_MANIFEST.json').read_bytes())
    keep_or_write(recovery/'restore_saved_results.py',Path(__file__).read_bytes())
    note='''# Восстановление после удаления временной среды Colab

Оба обученных адаптера, метаданные обучения и оригинальные baseline/prompt-only validation-журналы восстановлены из checkpoint_both_adapters.zip. Повторное обучение и новая генерация при восстановлении не выполнялись.

Заполненный REVIEW.md и два outputs_type*.txt возвращены из ранее сохранённых файлов без изменения текстов и оценок. Сохранённые baseline/prompt-only ответы и исходники сверены с оригинальными журналами checkpoint.

artifacts/adapted_validation.jsonl восстановлен из дословных ответов и диагностических подсказок в REVIEW.md; исходники дополнительно сверены с data/processed. Поле text получено применением прежнего formatter к сохранённому raw_text.

artifacts/adapted_test.jsonl — восстановленная таблица исходников и финальных постов из двух TXT. Оригинальный сырой test-ответ до formatter, исходные диагностические подсказки, токены, времена и finish_reason не восстановлены. Неизвестные поля имеют null; они не заменены вымышленными значениями. Эти JSONL не являются исходными полными журналами инференса.

Сводка оценок пересчитана из сохранённого REVIEW.md. Метаданные обучения остаются оригинальными. Пакеты текущей восстановленной среды не выдаются за окружение исходного обучения; версии исходных обучений сохранены в artifacts/train_type1.json и train_type2.json.
'''
    keep_or_write(root/'RECOVERY.md',note.encode('utf-8'))
    # Keep neutral assessment wording, and make generated reports disclose missing logs.
    patches={}
    f=root/'review_scores.py';code=f.read_text(encoding='utf-8')
    neutral=re.sub(r'Оценки поставлены[^.]*\.', 'Редакторские оценки даны относительно предоставленных исходников.',code)
    neutral=neutral.replace('# Сводная ручная оценка','# Сводная редакторская оценка')
    patches[f]=neutral
    f=root/'make_results.py';code=f.read_text(encoding='utf-8')
    old='Необработанные ответы, конфигурация генерации, длины и диагностические подсказки находятся в artifacts/adapted_test.jsonl.'
    new='Финальные посты восстановлены из сохранённых TXT в artifacts/adapted_test.jsonl; исходные сырые test-ответы, диагностические подсказки и замеры инференса утрачены. Подробности восстановления — RECOVERY.md.'
    if old in code:code=code.replace(old,new)
    elif new not in code:raise ValueError('Неизвестная версия make_results.py; требуется проверка перед изменением отчёта')
    if '## Восстановление после сброса Colab' not in code:
        needle='    instruction = '
        if needle not in code:raise ValueError('Не найдена точка добавления примечания в отчёт')
        code=code.replace(needle,'    md.extend(["", "## Восстановление после сброса Colab", "", (ROOT / "RECOVERY.md").read_text(encoding="utf-8")])\n'+needle,1)
    code=code.replace('Статус полной ручной проверки DATA_REVIEW.md','Статус проверки DATA_REVIEW.md')
    code=code.replace('Ручной вывод о фактах, стиле и структуре','Вывод о фактах, стиле и структуре')
    code=code.replace('Статус ручной проверки всех десяти постов','Статус проверки всех десяти постов')
    patches[f]=code
    f=root/'finalize_submission.py';code=f.read_text(encoding='utf-8')
    code=code.replace('Ручная оценка трёх режимов','Редакторская оценка трёх режимов')
    code=code.replace('Точные параметры, промпты, версии и сырые генерации — в `artifacts/`.',
        'Параметры, версии и сохранённые либо восстановленные записи — в `artifacts/`. После сброса Colab часть текстовых записей восстановлена из REVIEW.md и TXT; утраченная телеметрия не восстановлена. Подробности — в `RECOVERY.md`.')
    patches[f]=code
    for f,code in patches.items():
        compile(code,str(f),'exec')
    for f,code in patches.items():
        if f.read_text(encoding='utf-8')!=code:
            original=recovery/'original_scripts'/f.name
            if not original.exists():original.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,original)
            f.write_text(code,encoding='utf-8')
    result=subprocess.run([sys.executable,'-u','review_scores.py'],cwd=root,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8')
    print(result.stdout,flush=True);result.check_returncode()
    print('ВОССТАНОВЛЕНИЕ ЗАВЕРШЕНО. Нового обучения и генерации не было.')
    print('Созданы восстановленные записи и RECOVERY.md; исходные веса и логи checkpoint сохранены.')
    print('Следующий шаг: make_results.py, если RESULTS.md ещё не создан.')


if __name__=='__main__':main()
