import pdfplumber
import time

# --- PARAMÈTRES DES FICHIERS ---
fichier_original = "data/bnp-urd-2025-fr-mel3.pdf"
fichier_markdown = "data/bnp_risque_credit.md"

# Pages du chapitre Risque de Crédit (413 à 514)
page_debut = 413
page_fin = 514

print(f"📄 Extraction structurée avec pdfplumber (Pages {page_debut} à {page_fin})...")
start_time = time.time()

markdown_content = "# Extrait du Rapport de Risque - BNP Paribas\n\n"

with pdfplumber.open(fichier_original) as pdf:
    # L'index Python commence à 0, donc on retire 1
    for num_page in range(page_debut - 1, min(page_fin, len(pdf.pages))):
        page = pdf.pages[num_page]
        
        # --- 1. EXTRACTION DU TEXTE EN DOUBLE COLONNE ---
        largeur = page.width
        hauteur = page.height
        
        # Découpage virtuel de la page au centre
        boite_gauche = (0, 0, largeur / 2, hauteur)
        boite_droite = (largeur / 2, 0, largeur, hauteur)
        
        texte_gauche = page.within_bbox(boite_gauche).extract_text()
        texte_droite = page.within_bbox(boite_droite).extract_text()
        
        texte = ""
        if texte_gauche:
            texte += texte_gauche + "\n\n"
        if texte_droite:
            texte += texte_droite + "\n\n"
            
        markdown_content += f"\n\n## Page {num_page + 1}\n\n"
        if texte:
            markdown_content += texte
            
        # --- 2. EXTRACTION ET FORMATAGE DES TABLEAUX ---
        tables = page.extract_tables()
        for i, table in enumerate(tables):
            if table and len(table) > 0:
                markdown_content += f"\n\n### Tableau financier (Page {num_page + 1})\n"
                
                # En-tête du tableau
                markdown_content += "| " + " | ".join([str(cell).replace('\n', ' ') if cell else "" for cell in table[0]]) + " |\n"
                # Ligne de séparation Markdown
                markdown_content += "| " + " | ".join(["---"] * len(table[0])) + " |\n"
                # Contenu des lignes
                for row in table[1:]:
                    markdown_content += "| " + " | ".join([str(cell).replace('\n', ' ') if cell else "" for cell in row]) + " |\n"
                markdown_content += "\n---\n"

# Sauvegarde du fichier Markdown
with open(fichier_markdown, "w", encoding="utf-8") as f:
    f.write(markdown_content)

temps_ecoule = time.time() - start_time
print(f"✅ Extraction terminée en {temps_ecoule:.2f} secondes !")
print(f"📄 Fichier Markdown prêt : {fichier_markdown}")