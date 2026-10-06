import json
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

class DocumentExtractor:
    def __init__(self, llm_model: str = "mistral"):
        self.llm = Ollama(model=llm_model, format="json")
        self.prompt = PromptTemplate.from_template(
            "Tu es un expert financier. Extrais les 15 indicateurs financiers "
            "du texte fourni et renvoie UNIQUEMENT un objet JSON valide avec les cles exactes.\n"
            "Chaque valeur DOIT ETRE UN NOMBRE (float). N'ecris jamais de texte comme 'Not provided'. "
            "Si une donnee est introuvable, mets exactement 0.0.\n"
            "Cles obligatoires (garde bien l'espace au debut de la cle !) :\n"
            "\" Operating Profit Rate\", \" Research and development expense rate\", "
            "\" Interest-bearing debt interest rate\", \" Tax rate (A)\", "
            "\" After-tax Net Profit Growth Rate\", \" Net Value Growth Rate\", "
            "\" Quick Ratio\", \" Borrowing dependency\", \" Total Asset Turnover\", "
            "\" Revenue per person\", \" Allocation rate per person\", "
            "\" Quick Assets/Total Assets\", \" Cash/Total Assets\", "
            "\" Current Liabilities/Liability\", \" Liability-Assets Flag\"\n\n"
            "Texte financier :\n{text}\n\n"
            "JSON (nombres uniquement) :"
        )

    def extract(self, text: str) -> dict:
        chain = self.prompt | self.llm
        raw_response = chain.invoke({"text": text})
        try:
            data = json.loads(raw_response)
            # Nettoyage agressif : forcer tout en float
            clean_data = {}
            for k, v in data.items():
                try:
                    clean_data[k] = float(v)
                except:
                    clean_data[k] = 0.0
            return clean_data
        except Exception as e:
            raise ValueError(f"L'IA n'a pas retourne un JSON valide : {raw_response}")
