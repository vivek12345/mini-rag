from chonkie import RecursiveChunker

class Chunker:
    def __init__(self, lang: str = "en"):
        self.chunker = RecursiveChunker.from_recipe("markdown", lang=lang)

    def chunk(self, text: str):
        return self.chunker.chunk(text)

if __name__ == "__main__":
    from mini.loader import DocumentLoader
    loader = DocumentLoader()
    document = loader.load("./mini/documents/eb_test.pdf")
    chunker = Chunker()
    chunks = chunker.chunk(document)
    for chunk in chunks:
        print(chunk)