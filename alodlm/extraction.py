"""Answer extraction. Run: python -m unittest discover -s tests.

Copyright 2025 Tencent wechat. All rights reserved.
Modified from WeDLM evaluators; see optimized/licenses/WeDLM.txt.
"""
import re
from typing import List, Optional

class ARCCEvaluator:

    def _extract_answer(self, generation: str, available_options: List[str]) -> str:
        if not generation:
            return ''
        text = generation.strip()
        valid_options_set = {opt.upper() for opt in available_options}
        direct = re.fullmatch('\\(?\\s*([A-Za-z])\\s*\\)?[.!]?', text)
        if direct and direct.group(1).upper() in valid_options_set:
            return direct.group(1).upper()
        letters = re.escape(''.join(sorted(valid_options_set)))
        if not letters:
            return ''
        cue = f'(?:answer|Answer|ANSWER)\\s*(?:is|:)?\\s*\\(?\\s*([{letters}])\\b'
        ms = list(re.finditer(cue, text))
        if ms:
            candidate = ms[-1].group(1).upper()
            if candidate in valid_options_set:
                return candidate
        patterns = [f'^([{letters}])[.\\s)]', f'(?:answer|Answer|ANSWER)[\\s:]*([{letters}])', f'(?:is|IS)[\\s:]*([{letters}])']
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                candidate = match.group(1).upper()
                if candidate in valid_options_set:
                    return candidate
        ms = list(re.finditer(f'\\b([{letters}])\\b', text))
        for m in reversed(ms):
            candidate = m.group(1).upper()
            if candidate in valid_options_set:
                return candidate
        return ''

class MMLUEvaluator:

    def _extract_answer(self, generation: str) -> str:
        if not generation:
            return ''
        text = generation
        tail = text.rsplit('</think>', 1)[1] if '</think>' in text else text
        for scope in (tail, text):
            ms = list(re.finditer('(?:answer|Answer|ANSWER)\\s*(?:is|:)?\\s*\\(?\\s*([A-D])\\b', scope))
            if ms:
                return ms[-1].group(1).upper()
        mb = list(re.finditer('\\\\boxed\\{\\(?([A-D])\\)?\\}', text))
        if mb:
            return mb[-1].group(1).upper()
        for scope in (tail, text):
            ms = list(re.finditer('\\(([A-D])\\)', scope)) or list(re.finditer('\\b([A-D])\\b', scope))
            if ms:
                return ms[-1].group(1).upper()
        return ''

class GPQAEvaluator:

    def _extract_answer(self, generation: str) -> str:
        import re
        if not generation:
            return ''
        text = generation
        tail = text.rsplit('</think>', 1)[1] if '</think>' in text else text
        for scope in (tail, text):
            ms = list(re.finditer('(?:answer|Answer|ANSWER)\\s*(?:is|:)?\\s*\\(?\\s*([A-D])\\b', scope))
            if ms:
                return ms[-1].group(1).upper()
        mb = list(re.finditer('####\\s*\\(?([A-D])\\)?', text)) or list(re.finditer('\\\\boxed\\{\\(?([A-D])\\)?\\}', text))
        if mb:
            return mb[-1].group(1).upper()
        for scope in (tail, text):
            ms = list(re.finditer('\\(([A-D])\\)', scope)) or list(re.finditer('\\b([A-D])\\b', scope))
            if ms:
                return ms[-1].group(1).upper()
        return ''

def _last_boxed_only_string(string: str) -> Optional[str]:
    idx = string.rfind('\\boxed')
    if idx < 0:
        idx = string.rfind('\\fbox')
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == '{':
            num_left_braces_open += 1
        if string[i] == '}':
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx is None:
        return None
    else:
        return string[idx:right_brace_idx + 1]

def _remove_boxed(s: str) -> Optional[str]:
    left = '\\boxed{'
    try:
        if s.startswith(left):
            count = 0
            start = len(left) - 1
            for i in range(start, len(s)):
                if s[i] == '{':
                    count += 1
                elif s[i] == '}':
                    count -= 1
                    if count == 0:
                        return s[len(left):i]
            return None
        left = '\\fbox{'
        if s.startswith(left):
            count = 0
            start = len(left) - 1
            for i in range(start, len(s)):
                if s[i] == '{':
                    count += 1
                elif s[i] == '}':
                    count -= 1
                    if count == 0:
                        return s[len(left):i]
            return None
        return None
    except Exception:
        return None

def _strip_string(string: str) -> str:
    if not isinstance(string, str):
        return ''
    string = string.replace('\n', '').replace('\\!', '')
    string = string.replace('\\\\', '\\')
    string = string.replace('tfrac', 'frac').replace('dfrac', 'frac')
    string = string.replace('\\left', '').replace('\\right', '')
    string = string.replace('^{\\circ}', '').replace('^\\circ', '')
    string = string.replace('\\$', '').replace('$', '')
    string = string.replace('\\%', '').replace('%', '')
    string = string.replace(' ', '')
    string = re.sub('(\\d),(\\d)', '\\1\\2', string)
    while string.startswith('{') and string.endswith('}'):
        count = 0
        is_single_pair = True
        for i, c in enumerate(string):
            if c == '{':
                count += 1
            elif c == '}':
                count -= 1
                if count == 0 and i < len(string) - 1:
                    is_single_pair = False
                    break
        if is_single_pair and count == 0:
            string = string[1:-1]
        else:
            break
    return string

class MATHEvaluator:

    def _truncate_extra_problems(self, generation: str) -> str:
        if not generation:
            return ''
        patterns = ['\nProblem:', '\n\nProblem:', '\nProblem ', '\n\nProblem ']
        min_idx = len(generation)
        for pattern in patterns:
            idx = generation.find(pattern)
            if idx > 0:
                min_idx = min(min_idx, idx)
        return generation[:min_idx] if min_idx < len(generation) else generation

    def _fallback_extract(self, generation: str) -> str:
        if not generation:
            return ''
        text = generation
        if '</think>' in text:
            text = text.rsplit('</think>', 1)[1]
        ms = list(re.finditer('answer\\s+is\\s*:?\\s*\\$?([^\\n$]+?)\\$?\\s*(?:\\.\\s|\\.$|\\n|$)', text, re.IGNORECASE))
        if ms:
            cand = ms[-1].group(1).strip().rstrip('.')
            if cand:
                return cand
        dollars = re.findall('\\$([^$]+)\\$', text)
        if dollars:
            return dollars[-1].strip()
        nums = re.findall('-?\\d+(?:\\.\\d+)?', text)
        if nums:
            return nums[-1]
        return ''

    def _extract_answer(self, generation: str) -> str:
        if not generation:
            return ''
        generation = self._truncate_extra_problems(generation)
        boxed_str = _last_boxed_only_string(generation)
        answer = _remove_boxed(boxed_str) if boxed_str is not None else None
        if answer is None:
            return self._fallback_extract(generation)
        while answer.startswith('{') and answer.endswith('}'):
            count = 0
            is_single_wrapped_pair = True
            for i, char in enumerate(answer[:-1]):
                if char == '{':
                    count += 1
                elif char == '}':
                    count -= 1
                if count == 0:
                    is_single_wrapped_pair = False
                    break
            if is_single_wrapped_pair:
                answer = answer[1:-1]
            else:
                break
        return answer

    def _is_equiv(self, pred_str: str, gold_str: str) -> bool:
        if pred_str is None or gold_str is None:
            return False
        pred_norm = _strip_string(pred_str)
        gold_norm = _strip_string(gold_str)
        return pred_norm == gold_norm
