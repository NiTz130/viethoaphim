from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = ""
    openai_base_url: str = ""
    llm_model: str = ""
    edge_voice: str = "vi-VN-HoaiMyNeural"
    stt_model: str = "medium"
    stt_language: str = "zh"
    jobs_dir: str = "jobs"
    sample_rate: int = 44_100
