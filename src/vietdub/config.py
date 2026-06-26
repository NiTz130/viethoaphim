from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = ""
    anthropic_base_url: str = "https://api.minimax.io/anthropic"
    llm_model: str = "MiniMax-M3"

    tts_voice_id: str = "vi-VN-HoaiMyNeural"
    tts_model: str = "speech-2.8-hd"
    tts_base_url: str = "https://api.minimax.io/v1/t2a_v2"
    tts_api_key: str = ""
    tts_timeout: float = 30.0

    stt_model: str = "medium"
    stt_language: str = "zh"
    jobs_dir: str = "jobs"
    reference_data_dir: str = "data"
    sample_rate: int = 44_100
