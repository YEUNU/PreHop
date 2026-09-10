#!/usr/bin/env python3
"""Build the ACL review PDF from docs/prehop_paper.md without touching experiments.

Requires pypandoc_binary (or pypandoc and pandoc) and a Tectonic executable.
Official ACL style files and bibliography live in docs/acl. Generated TeX is
retained there; compilation intermediates go to a temporary directory.
"""
import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pypandoc

ROOT = Path(__file__).resolve().parents[1]
ACL = ROOT / 'docs' / 'acl'
CITATIONS = {
    'Cormack et al., 2009': 'cormack2009', 'Edge et al., 2024': 'edge2024',
    'Gao et al., 2023': 'gao2023', 'Guo et al., 2025': 'guo2025',
    'Lewis et al., 2020': 'lewis2020', 'Liu et al., 2025': 'liu2025',
    'Luo et al., 2025': 'luo2025', 'Nogueira et al., 2019': 'nogueira2019',
    'Tang and Yang, 2024': 'tang2024', 'Yang et al., 2018': 'yang2018',
    'Trivedi et al., 2023': 'trivedi2023', 'Zhuang et al., 2026': 'zhuang2026',
}

def latex(s):
    # Normalize mathematical question roles outside fenced prompts (which are literal).
    s = s.replace('→', r'$\rightarrow$').replace('κ', r'$\kappa$')
    s = s.replace('Q−', r'$Q^{-}$').replace('Q+', r'$Q^{+}$')
    return pypandoc.convert_text(s, 'latex', format='markdown+raw_tex',
                                extra_args=['--wrap=none', '--syntax-highlighting=none']).strip()

def raw(s):
    return '\n```{=latex}\n' + s + '\n```\n'

def header(s, align='l'):
    breaks = {
        'HOP relevance (%)': r'HOP relevance\\(\%)',
        'Added coverage (pp)': r'Added\\coverage (pp)',
        'Retained coverage (pp)': r'Retained\\coverage (pp)',
        'Mean HOP destinations': r'Mean HOP\\destinations',
        'Connection latency (ms)': r'Connection\\latency (ms)',
        'Destination agreement (%)': r'Destination\\agreement (\%)',
        'Connection saving (ms), 95% CI': r'Connection saving\\(ms), 95\% CI',
        'Query saving (s), 95% CI': r'Query saving\\(s), 95\% CI',
        'Link preparation (s)': r'Link preparation\\(s)',
        'Sampled edges': r'Sampled\\edges',
        'Supported Q+ links (%)': r'Supported\\$Q^{+}$ links (\%)',
        'Meaningful transitions (%)': r'Meaningful\\transitions (\%)',
        'Unclear transitions': r'Unclear\\transitions',
        'Agreement (κ)': r'Agreement\\($\kappa$)',
        'Returned evidence: min / median / max': r'Returned evidence\\min / median / max',
        'Index wall (s)': r'Index wall\\(s)',
        'Mean query latency (s)': r'Mean query\\latency (s)',
        'Batch s/query': r'Batch\\s/query',
        'Answer EM (%)': r'Answer\\EM (\%)', 'Answer F1 (%)': r'Answer\\F1 (\%)',
        'Supporting Fact EM (%)': r'Supporting Fact\\EM (\%)',
        'Supporting Fact F1 (%)': r'Supporting Fact\\F1 (\%)',
        'Joint EM (%)': r'Joint\\EM (\%)', 'Joint F1 (%)': r'Joint\\F1 (\%)',
        'Recall@10 (%)†': r'Recall@10\\(\%)\textsuperscript{\dag}',
        'AllFacts@10 (%)†': r'AllFacts@10\\(\%)\textsuperscript{\dag}',
        'Hits@4 (%)': r'Hits@4\\(\%)', 'Hits@10 (%)': r'Hits@10\\(\%)',
        'MultiHop-RAG condition': 'Condition', 'HotpotQA fullwiki condition': 'Condition',
        'Median total words per question': r'Median total words\\per question',
        'Maximum output tokens': r'Maximum\\output tokens',
        '2 facts (n=1,079)': r'2 facts\\($n=1{,}079$)',
        '3 facts (n=778)': r'3 facts\\($n=778$)',
        '4 facts (n=398)': r'4 facts\\($n=398$)',
    }
    return r'\makecell[' + align + ']{' + breaks.get(s, latex(s)) + '}'

def table(m):
    rows = [[c.strip() for c in row.strip().strip('|').split('|')]
            for row in m[1].strip().splitlines()]
    number, caption = int(m[2]), m[3]
    wide = number in {1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 13}
    env, width = ('table*', r'\textwidth') if wide else ('table', r'\columnwidth')
    if number == 1:
        spec = r'@{}p{0.15\linewidth}X p{0.25\linewidth}@{}'
    elif number == 5:
        spec = r'@{}p{0.29\linewidth}p{0.26\linewidth}X l@{}'
    elif number in {8, 11}:
        spec = '@{}Xl' + 'r' * (len(rows[0])-2) + '@{}'
    elif number == 9:
        spec = r'@{}p{0.16\linewidth}p{0.17\linewidth}rrrrr@{}'
    else:
        spec = '@{}X' + 'r' * (len(rows[0])-1) + '@{}'
    lines = [r'\begin{' + env + '}[!t]', r'\centering', r'\small', r'\setlength{\tabcolsep}{3pt}',
             r'\begin{tabularx}{' + width + '}{' + spec + '}', r'\toprule']
    lines.append(' & '.join(header(c, 'l' if j == 0 or number in {1, 5} or (number in {8, 9, 11} and j == 1) else 'r') for j, c in enumerate(rows[0])) + r' \\')
    lines.append(r'\midrule')
    for row in rows[2:]:
        lines.append(' & '.join(latex(c) for c in row) + r' \\')
    lines += [r'\bottomrule', r'\end{tabularx}',
              r'\caption{' + latex(caption) + '}', r'\label{tab:' + str(number) + '}',
              r'\end{' + env + '}']
    return raw('\n'.join(lines))

def figure(m):
    path, number, caption = m[1], m[2], m[3]
    pdf = (ROOT/'docs'/path).resolve().with_suffix('.pdf')
    if not pdf.is_file():
        raise FileNotFoundError(pdf)
    return raw(r'\begin{figure*}[!t]' + '\n' + r'\centering' + '\n' +
               r'\includegraphics[width=\textwidth]{../../fig/' + pdf.name + '}\n' +
               r'\caption{' + latex(caption) + '}\n' + r'\label{fig:' + number + '}\n' +
               r'\end{figure*}')

def generate():
    text = (ROOT/'docs/prehop_paper.md').read_text()
    title, text = text.split('\n', 1)
    title = title.removeprefix('# ')
    # Source references remain readable in Markdown; BibTeX controls PDF ordering/style.
    text = re.sub(r'## References\n.*?(?=## Appendix A\.)',
                  lambda _: raw(r'\FloatBarrier' + '\n' + r'\bibliography{references}' + '\n' + r'\clearpage\appendix'),
                  text, flags=re.DOTALL)
    for author, key in CITATIONS.items():
        text = text.replace('(' + author + ')', r'\citep{' + key + '}')
    text = re.sub(r'(!\[[^\]]*\]\(([^)]+)\))\n\n\*\*Figure (\d+)\.\*\* (.*?)(?=\n\n|\Z)',
                  lambda m: figure((None, m[2], m[3], m[4])), text, flags=re.DOTALL)
    text = re.sub(r'(\|[^\n]+\n(?:\|[^\n]+\n)+)\n\*\*Table (\d+)\.\*\* (.*?)(?=\n\n|\Z)',
                  table, text, flags=re.DOTALL)
    text = re.sub(r'^## Appendix [A-Z]\. (.*)$', r'## \1', text, flags=re.MULTILINE)
    text = re.sub(r'^(#{2,3}) (?:\d+\.|[A-Z]\.\d+\.|\d+\.\d+\.) (.*)$',
                  r'\1 \2', text, flags=re.MULTILINE)
    text = text.replace('## Abstract\n', raw(r'\begin{abstract}'), 1)
    text = text.replace('## Introduction', raw(r'\end{abstract}') + '\n## Introduction', 1)
    text = text.replace('## Discussion', raw(r'\FloatBarrier') + '\n## Discussion')
    text = text.replace('## Conclusion', raw(r'\Needspace{12\baselineskip}') + '\n## Conclusion')
    text = text.replace('## Limitations', raw(r'\FloatBarrier\clearpage\section*{Limitations}'))
    # Replace roles only in normal text, leaving all fenced prompt text literal.
    pieces = re.split(r'(```.*?```)', text, flags=re.DOTALL)
    for i in range(0, len(pieces), 2):
        pieces[i] = pieces[i].replace('Q−', r'$Q^{-}$').replace('Q+', r'$Q^{+}$')
    text = ''.join(pieces)
    body = pypandoc.convert_text(text, 'latex', format='markdown+raw_tex',
                                 extra_args=['--wrap=none', '--syntax-highlighting=none', '--shift-heading-level-by=-1'])
    preamble = r'''% Generated from docs/prehop_paper.md by scripts/render_acl_paper.py.
% Edit the Markdown source, not this generated file.
\documentclass[11pt]{article}
\usepackage[review]{acl}
\usepackage{times}
\usepackage{latexsym}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{microtype}
\usepackage{inconsolata}
\usepackage{graphicx}
\usepackage{amsmath,amssymb}
\usepackage{booktabs,tabularx,makecell}
\usepackage{placeins,needspace}
\usepackage{fvextra}
\usepackage{xurl}
\font\prehopmonofont="DejaVu Sans Mono" at 10pt
\DefineVerbatimEnvironment{verbatim}{Verbatim}{breaklines=true,breakanywhere=true,fontsize=\small,formatcom=\prehopmonofont}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
'''
    tex = preamble + '\n' + r'\title{' + latex(title) + '}\n' + r'\author{Anonymous ACL submission}' + '\n' + r'\begin{document}\raggedbottom\maketitle' + '\n' + body + '\n' + r'\end{document}' + '\n'
    output = ACL/'prehop_paper.tex'
    output.write_text(tex.replace('†', r'\textdagger{}'))
    return output

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tectonic', default='tectonic')
    parser.add_argument('--tex-only', action='store_true')
    args = parser.parse_args()
    tex = generate()
    print(f'Generated {tex}', flush=True)
    if args.tex_only:
        return
    build = Path(tempfile.mkdtemp(prefix='prehop-acl-build-'))
    subprocess.run([args.tectonic, '--keep-logs', '--keep-intermediates',
                    '--outdir', str(build), str(tex)], check=True, cwd=ACL)
    shutil.copy2(build/'prehop_paper.pdf', ROOT/'prehop_paper.pdf')
    print(f'PDF: {ROOT / "prehop_paper.pdf"}\nBuild log: {build / "prehop_paper.log"}')

if __name__ == '__main__':
    main()
