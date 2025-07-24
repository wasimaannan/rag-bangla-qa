from app.embed_store import build_vector_store
from app.qa_system import answer_question

# Uncomment to build vector DB once
# build_vector_store()

question = "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"
print("Answer:", answer_question(question))
