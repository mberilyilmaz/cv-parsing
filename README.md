# Resume Parsing System (Starter)

## Kurulum

1. Sanal ortam olustur:
   - `python -m venv .venv`
   - Windows: `.venv\\Scripts\\activate`
2. Bagimliliklari yukle:
   - `pip install -r requirements.txt`

## Calistirma

- Streamlit uygulamasi:
  - `streamlit run app/streamlit_app.py`

## Ilk Surum Kapsami

- PDF metin cikarma (pdfminer + PyPDF2 fallback)
- On isleme (cleaning, tokenization, stopword)
- Contact extraction (email/phone/linkedin/github)
- Rule-based skill extraction
- SQLite kayit
- Plotly skill frekans grafigi

## Sonraki Adim

- spaCy custom NER egitimi
- Section-based extraction (education/experience/projects)
- Degerlendirme (precision/recall/f1)
