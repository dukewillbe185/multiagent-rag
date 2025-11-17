from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
from dotenv import load_dotenv
import os

load_dotenv()

endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT").rstrip('/')
key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")

print(f"Testing endpoint: {endpoint}")
print(f"SDK: azure-ai-formrecognizer (stable)")

try:
    client = DocumentAnalysisClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key)
    )
    
    print("✅ Client created successfully")
    
    # 分析文档
    with open("doc.pdf", "rb") as f:
        print("📄 Analyzing document...")
        poller = client.begin_analyze_document("prebuilt-layout", document=f)
        
        print("⏳ Waiting for analysis...")
        result = poller.result()
    
    print("✅ Success!")
    print(f"📊 Pages: {len(result.pages)}")
    
    # 提取一些文本
    if result.content:
        print(f"📝 Text length: {len(result.content)} characters")
        print(f"📝 First 200 chars: {result.content[:200]}...")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print(f"Error type: {type(e).__name__}")