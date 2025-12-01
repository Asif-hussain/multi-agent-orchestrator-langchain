#!/usr/bin/env python3
"""
Chunk Count Verification Script
Verifies that each domain has minimum 50 chunks as required by assignment
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get chunk settings from environment
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

print("="*80)
print("DOCUMENT CHUNK VERIFICATION")
print("="*80)
print(f"\nConfiguration:")
print(f"  CHUNK_SIZE: {CHUNK_SIZE}")
print(f"  CHUNK_OVERLAP: {CHUNK_OVERLAP}")
print()

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# Domains to check
domains = {
    'HR': 'data/hr_docs',
    'IT/Tech': 'data/tech_docs',
    'Finance': 'data/finance_docs'
}

results = {}
all_passed = True

for domain_name, path in domains.items():
    print(f"\n{domain_name} Documents:")
    print("-" * 40)

    # Load documents
    loader = DirectoryLoader(
        path,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'}
    )
    documents = loader.load()

    # Split into chunks
    chunks = text_splitter.split_documents(documents)

    # Store results
    results[domain_name] = {
        'files': len(documents),
        'chunks': len(chunks),
        'passed': len(chunks) >= 50
    }

    # Display results
    print(f"  Files: {len(documents)}")
    print(f"  Chunks: {len(chunks)}")
    print(f"  Status: {'PASS' if len(chunks) >= 50 else 'FAIL'} (requirement: 50+ chunks)")

    if len(chunks) < 50:
        all_passed = False
        shortage = 50 - len(chunks)
        print(f"  Shortage: {shortage} chunks")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print()

for domain_name, result in results.items():
    status = "PASS" if result['passed'] else "FAIL"
    print(f"{domain_name:15} {result['files']} files  →  {result['chunks']:3d} chunks  [{status}]")

print()
print("-" * 80)
if all_passed:
    print("RESULT: ALL DOMAINS MEET REQUIREMENT (50+ chunks)")
    print("Assignment requirement satisfied!")
else:
    print("RESULT: SOME DOMAINS DO NOT MEET REQUIREMENT")
    print("Action needed: Add more documents or reduce chunk size")
print("="*80)
