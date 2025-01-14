from flask import Flask, request, render_template
import pymupdf as fitz
import torch
import nltk
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import (
    pipeline, 
    AutoModelForSeq2SeqLM, 
    AutoTokenizer, 
    AutoModelForCausalLM
)
from typing import List, Dict

app = Flask(__name__)


# =========================
# (A) 加载模型
# =========================

# 1. QG (Question Generation) 模型
qg_model_name = "gq"  # 替换为你的 QG 模型
qg_tokenizer = AutoTokenizer.from_pretrained(qg_model_name)
qg_model = AutoModelForSeq2SeqLM.from_pretrained(qg_model_name)

# 2. QA (抽取式问答) 模型
qa_model_name = "ga"  # 替换为你的 PDF 抽取答案模型
qa_pipeline = pipeline("question-answering", model=qa_model_name, tokenizer=qa_model_name)

# 3. LLM (生成式回答，比如 Llama)
llama_model_path = "Llama3.2-1B"  # 替换为你的 Llama 模型路径
llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_path, use_fast=False)
llama_model = AutoModelForCausalLM.from_pretrained(
    llama_model_path,
    torch_dtype=torch.float16,
    device_map="cpu"
)

# print(llama_model.hf_device_map)

# 4. Sentence-Transformers (检索、评估)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# =========================
# (B) 业务逻辑辅助函数
# =========================

def chunk_text(raw_text, chunk_size=500, overlap=50):
    """
    将整段文本按指定大小进行切分，并留有 overlap 的重叠防止信息被硬切。
    """
    text = raw_text.strip()
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += (chunk_size - overlap)
    return chunks

def evaluate_questions_and_answers(
    pdf_content: str,
    questions: List[str],
    answers: List[str]
) -> Dict:
    results = {}

    # (1) 多样性
    results["question_diversity"] = question_diversity_score(questions, embedding_model)

    # (2) 覆盖度
    coverage_scores = [keyword_coverage_score(pdf_content, q) for q in questions]
    results["avg_question_coverage"] = float(np.mean(coverage_scores))

    # (3) 正确性
    ref_text = pdf_content[:500] if len(pdf_content) > 500 else pdf_content
    correctness_scores = []
    for ans in answers:
        correctness_scores.append(semantic_similarity(ref_text, ans, embedding_model))
    results["avg_answer_correctness"] = float(np.mean(correctness_scores))

    # (4) 额外指标：比如“平均答案长度”
    lengths = [len(a.split()) for a in answers]
    results["avg_answer_length"] = float(np.mean(lengths))

    # (5) 额外指标：比如“可回答性”
    # 可能需要先看 answers 是否是 "Error ..."，计数
    unanswerable_count = sum(1 for a in answers if a.lower().startswith("error") or len(a.strip()) < 5)
    results["unanswerable_ratio"] = unanswerable_count / len(answers)

    return {k: round(v, 4) for k,v in results.items()}

def generate_questions_for_paragraph(paragraph, each_paragraph_q=2):
    """
    使用 QG 模型一次生成多条问题，若输出包含 <sep>，则进行拆分。
    这里用随机采样参数来提高问题多样性。
    """
    prompt = f"Generate questions: {paragraph}"
    # 你可以根据业务需求调高temperature，增大top_p，以获得更随机的结果
    inputs = qg_tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = qg_model.generate(
        inputs,
        max_length=128,
        do_sample=True,       # 打开随机采样
        top_k=50,
        top_p=0.9,
        temperature=1.1,
        num_beams=1,          # 不用beam search
        num_return_sequences=each_paragraph_q,
        early_stopping=True
    )

    all_questions = []
    for output in outputs:
        raw_text = qg_tokenizer.decode(output, skip_special_tokens=True)
        splitted = raw_text.split("<sep>")
        splitted = [q.strip() for q in splitted if q.strip()]
        all_questions.extend(splitted)
    return all_questions

def answer_question_extractive(question, context_text):
    """
    调用抽取式 QA 模型，获取答案
    """
    try:
        res = qa_pipeline({"question": question, "context": context_text})
        return res["answer"]
    except Exception as e:
        return f"Error extracting answer from PDF: {str(e)}"

def answer_question_with_llm(question, context):
    """
    生成式回答
    """
    # 这里用一个更详细的 Prompt，告诉模型要简洁回答
    prompt = f"""You are an AI assistant helping to answer questions based on the given context.
Please provide a concise answer (under 50 words) focusing on key information, 
and avoid repeating the entire context verbatim.

Context:
{context}

Question:
{question}

Answer:
"""

    try:
        # 将 max_length=128 改用 max_new_tokens=256
        # 且继续使用 do_sample/top_p/temperature 等
        inputs = llama_tokenizer.encode(prompt, return_tensors="pt", max_length=1024, truncation=True)
        outputs = llama_model.generate(
            inputs,
            max_new_tokens=256,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=1.0
        )
        
        return llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        return f"Error generating with Llama: {str(e)}"

def retrieve_top_k_paragraphs(question, paragraphs, paragraph_embeddings, k=2):
    """
    基于 Sentence-Transformers 向量相似度，检索前 k 个最相关段落
    """
    q_emb = embedding_model.encode(question, convert_to_tensor=True)
    cos_scores = util.cos_sim(q_emb, paragraph_embeddings)[0]
    top_results = torch.topk(cos_scores, k)
    top_idxs = top_results[1].cpu().numpy()
    retrieved = [paragraphs[idx] for idx in top_idxs]
    return retrieved


def double_check_answers(pdf_answer, llm_answer):
    """
    简单的对比：
    - 如果 PDF 答案长度足够，默认认为 PDF 答案可靠
    - 如果和 LLM 完全相同 => remark = "Same answer from both PDF and LLM"
    - 如果不同 => remark = "Discrepancy ..."
    返回: (final_answer, remark)
    但我们只用 remark，用来告诉前端差异情况。
    """
    pdf_ans_clean = pdf_answer.strip().lower()
    llm_ans_clean = llm_answer.strip().lower()

    if len(pdf_ans_clean) > 5 and not pdf_ans_clean.startswith("error"):
        # PDF answer 非空
        if pdf_ans_clean == llm_ans_clean:
            return pdf_answer, "Same answer from both PDF and LLM"
        else:
            return pdf_answer, "Discrepancy (the answer from PDF more reliable)"
    else:
        # PDF answer 无效
        return llm_answer, "Discrepancy (the answer from LLM more reliable)"



# =========================
# (C) 评估逻辑 (可选)
# =========================

def semantic_similarity(text1: str, text2: str, model) -> float:
    emb1 = model.encode([text1], convert_to_tensor=True)
    emb2 = model.encode([text2], convert_to_tensor=True)
    sim_score = util.cos_sim(emb1, emb2)
    return float(sim_score[0][0])

def keyword_coverage_score(content_text: str, generated_text: str, top_k=10) -> float:
    words = nltk.word_tokenize(content_text.lower())
    stopwords = set(nltk.corpus.stopwords.words('english'))
    freq_dict = {}
    for w in words:
        if w not in stopwords and w.isalpha():
            freq_dict[w] = freq_dict.get(w, 0) + 1
    sorted_freq = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
    top_keywords = [k for k, _ in sorted_freq[:top_k]]

    gen_words = set(nltk.word_tokenize(generated_text.lower()))
    matched_count = sum(1 for kw in top_keywords if kw in gen_words)
    coverage = matched_count / len(top_keywords) if top_keywords else 1
    return coverage

def question_diversity_score(questions: List[str], model) -> float:
    if not questions:
        return 0.0
    embeddings = model.encode(questions, convert_to_tensor=True)
    sim_matrix = util.cos_sim(embeddings, embeddings)
    n = len(questions)
    total_sim = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_sim += sim_matrix[i][j]
            count += 1
    avg_sim = float(total_sim / count) if count > 0 else 0.0
    diversity = 1 - avg_sim
    return max(0.0, min(1.0, diversity))

def evaluate_questions_and_answers(
    pdf_content: str,
    questions: List[str],
    answers: List[str]
) -> Dict:
    """
    简单评估: (1)多样性, (2)覆盖度, (3)正确性(与pdf前500字符对比)
    """
    if len(questions) != len(answers):
        raise ValueError("Mismatch between the number of questions and answers.")

    # 1) 多样性
    diversity = question_diversity_score(questions, embedding_model)

    # 2) 覆盖度
    coverage_scores = [keyword_coverage_score(pdf_content, q) for q in questions]
    avg_coverage = float(np.mean(coverage_scores)) if coverage_scores else 0.0

    # 3) 答案正确性
    ref_text = pdf_content[:500] if len(pdf_content) > 500 else pdf_content
    correctness_scores = []
    for ans in answers:
        score = semantic_similarity(ref_text, ans, embedding_model)
        correctness_scores.append(score)
    avg_correctness = float(np.mean(correctness_scores)) if correctness_scores else 0.0
    
    lengths = [len(a.split()) for a in answers]
    avg_answer_length = float(np.mean(lengths))

    unanswerable_count = sum(
        1 for a in answers if a.lower().startswith("error") or len(a.strip()) < 5
    )
    unanswerable_ratio = unanswerable_count / len(answers)

    return {
        "question_diversity": round(diversity, 4),
        "avg_question_coverage": round(avg_coverage, 4),
        "avg_answer_correctness": round(avg_correctness, 4),
        "avg_answer_length": round(avg_answer_length, 4),
        "unanswerable_ratio": round(unanswerable_ratio, 4)
    }


# =========================
# (D) Flask 路由
# =========================

@app.route('/')
def index():
    return render_template('upload.html')


@app.route('/generate-quiz/', methods=['POST'])
def generate_quiz():
    # 1) 获取表单信息
    if 'file' not in request.files or 'num_questions' not in request.form:
        return "Missing file or num_questions in form.", 400

    file = request.files['file']
    num_questions = request.form['num_questions']

    # 2) 校验题目数量
    try:
        num_questions = int(num_questions)
        if not (1 <= num_questions <= 20):
            raise ValueError
    except ValueError:
        return "Number of questions must be an integer between 1 and 20.", 400

    # 3) 校验文件
    if not file.filename.lower().endswith('.pdf'):
        return "Invalid file format, only PDF is allowed.", 400

    # 4) 提取 PDF 文本
    try:
        pdf_doc = fitz.open(stream=file.read(), filetype="pdf")
        raw_text = "".join(page.get_text() for page in pdf_doc)
        if not raw_text.strip():
            raise ValueError("PDF contains no extractable text.")
    except Exception as e:
        return f"Failed to process PDF: {e}", 400

    # 5) 分块 (chunk) 处理文本
    #    这里采用 chunk_size=500, overlap=50 的示例
    paragraphs = chunk_text(raw_text, chunk_size=500, overlap=50)
    # 也可添加去重/过滤逻辑

    if not paragraphs:
        return "No valid paragraph after chunking the text.", 400

    entire_context = "\n\n".join(paragraphs)

    # 构建段落 embedding 用于检索
    paragraph_embeddings = embedding_model.encode(paragraphs, convert_to_tensor=True)

    # 6) QG 生成问题
    collected_qp = []  # [(question, paragraph_idx), ...]
    question_count = 0
    for idx, para in enumerate(paragraphs):
        if question_count >= num_questions:
            break
        # 假设每个 chunk 我们只生成 2 条问题
        each_paragraph_q = min(num_questions - question_count, 2)
        qlist = generate_questions_for_paragraph(para, each_paragraph_q=each_paragraph_q)
        for q in qlist:
            if question_count < num_questions:
                collected_qp.append((q, idx))
                question_count += 1
            else:
                break

    if not collected_qp:
        return "No questions generated!", 400

    # 7) 对每个问题做检索 + QA + LLM
    qa_pairs = []
    for question, para_idx in collected_qp:
        # (a) 检索最相关段落
        retrieved_paras = retrieve_top_k_paragraphs(question, paragraphs, paragraph_embeddings, k=2)
        context_for_qa = "\n".join(retrieved_paras)

        # (b) 抽取式 QA
        pdf_answer = answer_question_extractive(question, context_for_qa)

        # (c) LLM 生成答案
        llm_answer = answer_question_with_llm(question, entire_context)

        # (d) 对比 remark
        #    注意: double_check_answers() 返回 (final_answer, remark)
        #    但我们只需要把 "pdf_answer", "llm_answer", 和 "remark" 存起来。
        final_answer, remark = double_check_answers(pdf_answer, llm_answer)

        # => 这里存成四元组 (question, pdf_answer, llm_answer, remark)
        qa_pairs.append((question, pdf_answer, llm_answer, remark))

    # 8) 评估逻辑 (可选)
    questions_for_eval = [p[0] for p in qa_pairs]
    # 这里我们拿 p[1] 即 pdf_answer 做评估，或者也可以拿 final_answer
    answers_for_eval = [p[1] for p in qa_pairs]
    evaluation_result = evaluate_questions_and_answers(entire_context, questions_for_eval, answers_for_eval)

    return render_template('results.html', qa_pairs=qa_pairs, eval_result=evaluation_result)

    # 9) 渲染结果
    return render_template('results.html', qa_pairs=qa_pairs, eval_result=evaluation_result)


# =========================
# (E) 启动 Flask
# =========================

if __name__ == '__main__':
    app.run(debug=True)