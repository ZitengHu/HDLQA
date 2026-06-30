

import xml.etree.ElementTree as ET
import json
import argparse
import random
from bs4 import BeautifulSoup

# ─────────────────────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────────────────────

TARGET_TAGS = {"verilog", "vhdl", "systemverilog", "hdl"}


# ─────────────────────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────────────────────

def html_to_text(html):
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text()
    return text.replace('\n', ' ').replace('\r', ' ').strip()


def parse_tags(tags_str):
    if not tags_str or not tags_str.startswith('|'):
        return []
    return [t for t in tags_str.split('|') if t]


def tag_match(tags_str):
    """检查帖子标签是否在目标标签集合中"""
    return bool(set(parse_tags(tags_str)) & TARGET_TAGS)


# ─────────────────────────────────────────────────────────────
# 主要处理函数
# ─────────────────────────────────────────────────────────────

def filter_posts(xml_file, output_file, preview_file, preview_size=100):
    """
    从XML文件筛选帖子并保存到JSONL文件
    """
    print(f"开始处理文件: {xml_file}")

    # 第一次遍历：收集所有符合条件的问题
    questions = {}
    total_posts = 0

    print("第一遍扫描：查找符合条件的问题...")
    for event, elem in ET.iterparse(xml_file, events=('end',)):
        if elem.tag != 'row':
            elem.clear()
            continue

        post_type = elem.get('PostTypeId')
        if post_type != '1':  # 只处理问题
            elem.clear()
            continue

        total_posts += 1
        if total_posts % 100000 == 0:
            print(f"已处理 {total_posts} 个帖子...")

        tags = elem.get('Tags', '')
        if tag_match(tags):
            question_id = elem.get('Id')
            questions[question_id] = {
                'Id': question_id,
                'Title': elem.get('Title', ''),
                'Body': elem.get('Body', ''),
                'Tags': tags,
                'Score': int(elem.get('Score', 0)),
                'AcceptedAnswerId': elem.get('AcceptedAnswerId', ''),
                'CreationDate': elem.get('CreationDate', ''),
                'ViewCount': int(elem.get('ViewCount', 0)),
                'AnswerCount': int(elem.get('AnswerCount', 0))
            }

        elem.clear()

    print(f"总共处理了 {total_posts} 个帖子，找到 {len(questions)} 个符合条件的问题")

    # 第二次遍历：收集这些问题的回答
    answers = {}

    print("第二遍扫描：查找相关回答...")
    for event, elem in ET.iterparse(xml_file, events=('end',)):
        if elem.tag != 'row':
            elem.clear()
            continue

        post_type = elem.get('PostTypeId')
        if post_type != '2':  # 只处理回答
            elem.clear()
            continue

        parent_id = elem.get('ParentId', '')
        if parent_id in questions:  # 只收集目标问题的回答
            answer_list = answers.setdefault(parent_id, [])
            answer_list.append({
                'Id': elem.get('Id'),
                'Body': elem.get('Body', ''),
                'Score': int(elem.get('Score', 0)),
                'OwnerUserId': elem.get('OwnerUserId', ''),
                'CreationDate': elem.get('CreationDate', '')
            })

        elem.clear()

    print(f"收集到 {len(answers)} 个问题的相关回答")

    # 组装最终结果
    results = []
    for qid, question in questions.items():
        accepted_answer_id = question['AcceptedAnswerId']
        if not accepted_answer_id:
            continue  # 只保留有被采纳回答的问题

        answer_list = answers.get(qid, [])
        accepted_answer = None
        for ans in answer_list:
            if ans['Id'] == accepted_answer_id:
                accepted_answer = ans
                break

        if not accepted_answer:
            continue  # 被采纳回答不存在

        # 创建结果项
        result_item = {
            'questionId': question['Id'],
            'title': question['Title'].strip(),
            'question': html_to_text(question['Body']),
            'questionMetadata': {
                'tags': parse_tags(question['Tags']),
                'score': question['Score'],
                'viewCount': question['ViewCount'],
                'matchType': 'tag'
            },
            'answers': [{
                'answerId': accepted_answer['Id'],
                'type': 'accepted',
                'score': accepted_answer['Score'],
                'body': html_to_text(accepted_answer['Body']),
                'creationDate': accepted_answer['CreationDate'],
                'ownerUserId': accepted_answer['OwnerUserId']
            }]
        }

        # 添加得分高于被采纳回答的其他回答
        higher_score_answers = [
            ans for ans in answer_list
            if ans['Id'] != accepted_answer_id and ans['Score'] > accepted_answer['Score']
        ]
        higher_score_answers.sort(key=lambda x: x['Score'], reverse=True)

        for ans in higher_score_answers[:3]:  # 最多添加3个更高分的回答
            result_item['answers'].append({
                'answerId': ans['Id'],
                'type': 'top_voted',
                'score': ans['Score'],
                'body': html_to_text(ans['Body']),
                'creationDate': ans['CreationDate'],
                'ownerUserId': ans['OwnerUserId']
            })

        results.append(result_item)

    print(f"最终筛选出 {len(results)} 个有效问答对")

    # 保存完整结果
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # 保存预览结果
    preview_results = random.sample(results, min(preview_size, len(results)))
    with open(preview_file, 'w', encoding='utf-8') as f:
        for item in preview_results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"完整数据已保存至: {output_file}")
    print(f"预览数据已保存至: {preview_file}")
    return results


# ─────────────────────────────────────────────────────────────
# 主程序入口
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='筛选Stack Overflow硬件相关帖子')
    parser.add_argument('--input', '-i', default=r"E:\迅雷下载\stackoverflow.com-Posts\Posts.xml",help='输入的Posts.xml文件路径')
    parser.add_argument('--output', '-o',  default=r"D:\pycharm\so\stack-eval-main\data\stack-verilog-eval-v3.jsonl",help='输出的完整JSONL文件路径')
    parser.add_argument('--preview', '-p', default=r"D:\pycharm\so\stack-eval-main\data\stack-tinyverilog-eval-v3.jsonl",help='输出的预览JSONL文件路径')
    parser.add_argument('--size', '-s', type=int, default=363, help='预览文件大小，默认100')

    args = parser.parse_args()

    random.seed(42)
    results = filter_posts(args.input, args.output, args.preview, args.size)

    print(f"\n筛选完成！")
    print(f"- 目标标签: {TARGET_TAGS}")
    print(f"- 总问题数: {len(results)}")
