"""
upload_to_hf.py — Upload local chroma_db directory to Hugging Face Datasets
"""
import os
from dotenv import load_dotenv
from huggingface_hub import HfApi

# 1. تحميل المتغيرات من ملف .env
load_dotenv()

# 📝 استبدل YOUR_HF_USERNAME باسم حسابك الحقيقي على Hugging Face
HF_REPO_ID = "mostafaeltaweel/data-analysis-chromadb"

# جلب المفتاح من ملف .env
HF_TOKEN = os.getenv("HF_TOKEN")

def upload_chroma():
    chroma_dir = "./chroma_db"

    # التحقق من وجود المجلد محلياً
    if not os.path.exists(chroma_dir):
        print(f"❌ خطأ: المجلد '{chroma_dir}' غير موجود محلياً.")
        return

    # التحقق من وجود المفتاح
    if not HF_TOKEN or HF_TOKEN.startswith("hf_your"):
        print("❌ خطأ: لم يتم العثور على HF_TOKEN في ملف .env أو أنه لا يزال القيمة الافتراضية.")
        return

    print("⏳ جاري رفع مجلد 'chroma_db' إلى Hugging Face...")
    try:
        api = HfApi()
        api.upload_folder(
            folder_path=chroma_dir,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            token=HF_TOKEN,
        )
        print("✅ تم رفع chroma_db بنجاح إلى Hugging Face!")
        print(f"🔗 الرابط: https://huggingface.co/datasets/{HF_REPO_ID}")
    except Exception as e:
        print(f"❌ فشلت عملية الرفع: {e}")

if __name__ == "__main__":
    upload_chroma()