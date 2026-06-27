"""
Factory tạo LLM và Embeddings cho 5 providers: openai, gemini, anthropic, ollama, openrouter.

Cách dùng:
    from utils.llm_factory import get_llm, get_embeddings

    llm        = get_llm()            # dùng PROVIDER từ .env
    embeddings = get_embeddings()     # dùng PROVIDER từ .env

    llm_gemini = get_llm("gemini")    # chỉ định provider cụ thể
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

from langchain_core.embeddings import Embeddings


class _FastEmbedRagasSafe(Embeddings):
    """
    Proxy bọc FastEmbedEmbeddings để phơi thuộc tính `.model` dạng STRING.

    Lý do: RAGAS ghi telemetry EmbeddingUsageEvent với model=getattr(emb, "model"),
    trường này là Optional[str]. FastEmbedEmbeddings.model lại là object fastembed
    → pydantic ValidationError → metric answer_relevancy trả về NaN.
    Proxy ủy thác toàn bộ việc embed cho fastembed thật, nhưng .model là string
    (model_name) nên RAGAS không còn lỗi và answer_relevancy tính được bình thường.
    """

    def __init__(self, inner, model_name: str):
        self._inner = inner
        self.model = model_name          # string — RAGAS đọc getattr(emb, "model")
        self.model_name = model_name

    def embed_documents(self, texts):
        return self._inner.embed_documents(texts)

    def embed_query(self, text):
        return self._inner.embed_query(text)


def get_llm(provider: str = None, temperature: float = 0.0):
    """
    Trả về BaseChatModel tương ứng với provider được chọn.

    Args:
        provider    : "openai" | "gemini" | "anthropic" | "ollama" | "openrouter"
                      Mặc định: đọc PROVIDER từ .env (config.PROVIDER)
        temperature : độ ngẫu nhiên (0.0 = tất định, 1.0 = sáng tạo)

    Returns:
        BaseChatModel instance sẵn sàng sử dụng

    Raises:
        ValueError nếu provider không hợp lệ
        ImportError nếu package tương ứng chưa được cài đặt
    """
    provider = (provider or config.PROVIDER).lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        kwargs = {
            "model": config.OPENAI_MODEL,
            "api_key": config.OPENAI_API_KEY,
            "temperature": temperature,
        }
        if config.OPENAI_BASE_URL:
            kwargs["base_url"] = config.OPENAI_BASE_URL
        return ChatOpenAI(**kwargs)

    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
            temperature=temperature,
        )

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=config.ANTHROPIC_MODEL,
            api_key=config.ANTHROPIC_API_KEY,
            temperature=temperature,
        )

    elif provider == "ollama":
        # Dùng đường OpenAI-compatible của Ollama (/v1). RAGAS parse output ổn định
        # hơn nhiều so với ChatOllama (ChatOllama khiến RAGAS trả NaN do không parse được).
        from langchain_openai import ChatOpenAI
        base = config.OLLAMA_BASE_URL.rstrip("/")
        base = base if base.endswith("/v1") else base + "/v1"
        return ChatOpenAI(
            model=config.OLLAMA_MODEL,
            base_url=base,
            api_key="ollama",
            temperature=temperature,
            timeout=600,
        )

    elif provider == "openrouter":
        # OpenRouter dùng OpenAI-compatible API
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.OPENROUTER_MODEL,
            api_key=config.OPENROUTER_API_KEY,
            base_url=config.OPENROUTER_BASE_URL,
            temperature=temperature,
        )

    else:
        raise ValueError(
            f"Provider không hợp lệ: '{provider}'. "
            "Chọn một trong: openai, gemini, anthropic, ollama, openrouter"
        )


def get_embeddings(provider: str = None):
    """
    Trả về Embeddings instance tương ứng với provider được chọn.

    Lưu ý quan trọng:
        - Anthropic KHÔNG có Embeddings API → tự động fallback về OpenAI embeddings
        - OpenRouter cũng dùng OpenAI embeddings (không có API embeddings riêng)
        - Ollama cần model embedding riêng (mặc định: nomic-embed-text)
          Cài đặt: ollama pull nomic-embed-text

    Args:
        provider: "openai" | "gemini" | "anthropic" | "ollama" | "openrouter"
                  Mặc định: đọc PROVIDER từ .env

    Returns:
        Embeddings instance sẵn sàng sử dụng
    """
    provider = (provider or config.PROVIDER).lower()

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        kwargs = {
            "model": config.OPENAI_EMBEDDING_MODEL,
            "api_key": config.OPENAI_API_KEY,
        }
        if config.OPENAI_BASE_URL:
            kwargs["base_url"] = config.OPENAI_BASE_URL
        return OpenAIEmbeddings(**kwargs)

    elif provider == "openrouter":
        # OpenRouter KHÔNG có Embeddings API → dùng embeddings chạy local (fastembed),
        # miễn phí, không cần OpenAI key. Model mặc định: BAAI/bge-small-en-v1.5 (384-dim).
        # Bọc trong proxy để .model là string (tránh lỗi telemetry RAGAS → NaN answer_relevancy).
        print("ℹ️  OpenRouter không có Embeddings API — dùng FastEmbed (local, miễn phí).")
        from langchain_community.embeddings import FastEmbedEmbeddings
        inner = FastEmbedEmbeddings()
        return _FastEmbedRagasSafe(inner, getattr(inner, "model_name", "fastembed-local"))

    elif provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            model=config.GEMINI_EMBEDDING_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
        )

    elif provider == "anthropic":
        # Anthropic không cung cấp Embeddings API → dùng OpenAI thay thế
        print("⚠️  Anthropic không có Embeddings API — đang dùng OpenAI embeddings thay thế.")
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=config.OPENAI_EMBEDDING_MODEL,
            api_key=config.OPENAI_API_KEY,
        )

    elif provider == "ollama":
        # Dùng embeddings local fastembed (giống openrouter): không cần pull thêm
        # model embedding cho Ollama, và đã được kiểm chứng chạy tốt với RAGAS.
        print("ℹ️  Ollama: dùng FastEmbed (local, miễn phí) cho embeddings.")
        from langchain_community.embeddings import FastEmbedEmbeddings
        inner = FastEmbedEmbeddings()
        return _FastEmbedRagasSafe(inner, getattr(inner, "model_name", "fastembed-local"))

    else:
        raise ValueError(
            f"Provider không hợp lệ: '{provider}'. "
            "Chọn một trong: openai, gemini, anthropic, ollama, openrouter"
        )
