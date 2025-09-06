class DocumentLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.content = ""

    def read_document(self):
        if self.file_path.endswith('.pdf'):
            self.content = self._read_pdf()
        elif self.file_path.endswith('.docx'):
            self.content = self._read_docx()
        elif self.file_path.endswith('.txt'):
            self.content = self._read_txt()
        else:
            raise ValueError("Unsupported file format")

    def _read_pdf(self):
        import PyPDF2
        with open(self.file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page in reader.pages:
                text += page.extract_text() + '\n'
        return text

    def _read_docx(self):
        from docx import Document
        doc = Document(self.file_path)
        text = '\n'.join([para.text for para in doc.paragraphs])
        return text

    def _read_txt(self):
        with open(self.file_path, 'r', encoding='utf-8') as file:
            return file.read()

    def chunk_content(self, chunk_size=100):
        return [self.content[i:i + chunk_size] for i in range(0, len(self.content), chunk_size)]

    def process_document(self):
        self.read_document()
        return self.chunk_content()