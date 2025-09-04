import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

LIGNE_DEBUT_DONNEES = "______Output Data (Format for each row: index Number, scanning number, Waveform)______"
NOMBRE_PAS_TEMPS = 800 # Supposé constant, basé sur les informations précédentes

def analyser_fichier_donnees_dynamique(chemin_fichier):
    """
    Analyse le fichier de données pour extraire la matrice 3D,
    en déterminant dynamiquement le nombre de lignes et de colonnes.
    Cette version interprète le premier chiffre de chaque ligne de données comme l'indice de COLONNE
    et le deuxième chiffre comme l'indice de LIGNE.
    """
    donnees_brutes = []  # Pour stocker (idx_col_fichier, idx_ligne_fichier, waveform_array)
    max_observed_col_idx = -1 # Suit l'indice de colonne max lu (premier chiffre de la ligne de données)
    max_observed_row_idx = -1 # Suit l'indice de ligne max lu (deuxième chiffre de la ligne de données)
    
    en_tete_trouve = False
    lignes_donnees_valides_lues = 0
    
    print(f"Analyse dynamique du fichier : {chemin_fichier}")
    print("ATTENTION : Interprétation des indices de données : "
          "1er chiffre = INDICE DE COLONNE, 2ème chiffre = INDICE DE LIGNE.")
    print("Phase 1 : Lecture préliminaire pour déterminer les dimensions et stocker les données brutes...")

    try:
        with open(chemin_fichier, 'r', encoding='utf-8') as f:
            for num_ligne_fichier, ligne_contenu in enumerate(f, 1):
                ligne_nettoyee = ligne_contenu.strip()

                if not en_tete_trouve:
                    if ligne_nettoyee.startswith(LIGNE_DEBUT_DONNEES):
                        en_tete_trouve = True
                        print(f"En-tête '{LIGNE_DEBUT_DONNEES}' trouvé à la ligne {num_ligne_fichier}.")
                    continue

                if not ligne_nettoyee:
                    continue

                elements = ligne_nettoyee.split()
                
                if len(elements) != 2 + NOMBRE_PAS_TEMPS:
                    if en_tete_trouve:
                        print(f"Info (Ligne {num_ligne_fichier} - Phase 1): La ligne ne correspond pas au format attendu "
                              f"(2 indices + {NOMBRE_PAS_TEMPS} valeurs). "
                              f"Éléments trouvés: {len(elements)}. Contenu : '{ligne_nettoyee[:70]}...'. Ligne ignorée.")
                    continue

                try:
                    # Interprétation des indices selon la nouvelle spécification :
                    # elements[0] -> indice de colonne
                    # elements[1] -> indice de ligne
                    idx_col_fichier = int(elements[0])
                    idx_ligne_fichier = int(elements[1])
                    
                    waveform_str_values = elements[2:]
                    if len(waveform_str_values) == NOMBRE_PAS_TEMPS: # Redondant mais sûr
                        forme_onde = np.array([float(x) for x in waveform_str_values])
                        
                        # Stocker les indices tels qu'ils sont lus du fichier
                        donnees_brutes.append((idx_col_fichier, idx_ligne_fichier, forme_onde))
                        
                        # Mettre à jour les maximums observés
                        max_observed_col_idx = max(max_observed_col_idx, idx_col_fichier)
                        max_observed_row_idx = max(max_observed_row_idx, idx_ligne_fichier)
                        lignes_donnees_valides_lues += 1
                    # else: # Ce cas est couvert par la vérification len(elements) plus haut

                except ValueError:
                    if en_tete_trouve:
                        print(f"Avertissement (Ligne {num_ligne_fichier} - Phase 1): Impossible de convertir les indices "
                              f"ou valeurs de forme d'onde : '{ligne_nettoyee[:70]}...'. Ligne ignorée.")
    
    except FileNotFoundError:
        print(f"Erreur : Le fichier '{chemin_fichier}' n'a pas été trouvé.")
        return None
    except Exception as e:
        print(f"Une erreur inattendue est survenue lors de la lecture du fichier (Phase 1) : {e}")
        return None

    if not en_tete_trouve:
        print(f"L'en-tête '{LIGNE_DEBUT_DONNEES}' n'a pas été trouvé. Traitement annulé.")
        return None
        
    if not donnees_brutes:
        print("Aucune donnée valide n'a été lue après l'en-tête.")
        return None

    # Les dimensions de la matrice NumPy seront (nombre_de_lignes, nombre_de_colonnes, ...)
    # Le nombre de lignes est déterminé par max_observed_row_idx
    # Le nombre de colonnes est déterminé par max_observed_col_idx
    nombre_final_lignes_matrice = max_observed_row_idx + 1
    nombre_final_colonnes_matrice = max_observed_col_idx + 1
    
    print(f"Fin de la Phase 1 : {lignes_donnees_valides_lues} lignes de données valides lues.")
    print(f"Dimensions détectées pour la matrice NumPy finale : "
          f"{nombre_final_lignes_matrice} lignes (indice de ligne max lu du fichier: {max_observed_row_idx}), "
          f"{nombre_final_colonnes_matrice} colonnes (indice de colonne max lu du fichier: {max_observed_col_idx}).")
    print(f"Nombre de pas de temps utilisé : {NOMBRE_PAS_TEMPS}.")

    print("Phase 2 : Création de la matrice NumPy et remplissage des données...")
    try:
        if nombre_final_lignes_matrice <= 0 or nombre_final_colonnes_matrice <= 0:
            print(f"Erreur : Dimensions finales de la matrice non valides "
                  f"({nombre_final_lignes_matrice}x{nombre_final_colonnes_matrice}). Impossible de continuer.")
            return None

        donnees_matrice = np.zeros((nombre_final_lignes_matrice, nombre_final_colonnes_matrice, NOMBRE_PAS_TEMPS))
        
        lignes_remplies_matrice = 0
        for col_f, row_f, forme_onde_data in donnees_brutes:
            # Important: lors du remplissage de donnees_matrice[idx_ligne, idx_colonne, :],
            # utiliser row_f comme idx_ligne et col_f comme idx_colonne.
            if 0 <= row_f < nombre_final_lignes_matrice and 0 <= col_f < nombre_final_colonnes_matrice:
                donnees_matrice[row_f, col_f, :] = forme_onde_data
                lignes_remplies_matrice +=1
            else:
                print(f"Erreur interne critique (Phase 2): Indice fichier (col:{col_f}, row:{row_f}) "
                      f"hors des limites de la matrice ({nombre_final_colonnes_matrice}x{nombre_final_lignes_matrice}) "
                      "lors du remplissage. Donnée ignorée.")
        
        print(f"Matrice NumPy remplie. {lignes_remplies_matrice} points (ligne,colonne) "
              "ont été renseignés dans la matrice finale.")
        if lignes_remplies_matrice != lignes_donnees_valides_lues:
             print(f"Avertissement : {lignes_donnees_valides_lues} lignes de données lues, "
                   f"mais {lignes_remplies_matrice} insérées dans la matrice. Vérifiez les erreurs.")
        return donnees_matrice
        
    except ValueError as ve: 
        print(f"Erreur lors de la création de la matrice NumPy (Phase 2) : {ve}")
        return None
    except Exception as e:
        print(f"Une erreur inattendue est survenue lors de la création/remplissage de la matrice (Phase 2) : {e}")
        return None


def afficher_matrice_interactive(donnees_matrice):
    """
    Affiche la matrice interactivement avec un curseur pour le temps.
    S'adapte aux dimensions de la matrice fournie (lignes, colonnes, temps).
    """
    if donnees_matrice is None or donnees_matrice.size == 0:
        print("Aucune donnée à afficher ou matrice vide.")
        return

    num_lignes_mat = donnees_matrice.shape[0]
    num_colonnes_mat = donnees_matrice.shape[1]
    num_pas_temps_mat = donnees_matrice.shape[2]

    if num_pas_temps_mat == 0:
        print("La matrice ne contient aucun pas de temps. Impossible d'afficher.")
        return
    if num_lignes_mat == 0 or num_colonnes_mat == 0:
        print("La matrice a des dimensions de lignes ou de colonnes nulles. Impossible d'afficher.")
        return

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    temps_initial = 0
    img_affichee = ax.imshow(donnees_matrice[:, :, temps_initial], cmap='viridis', aspect='auto', origin='lower')
    
    ax.set_xlabel(f"Indice de Colonne (0 à {num_colonnes_mat - 1})")
    ax.set_ylabel(f"Indice de Ligne (0 à {num_lignes_mat - 1})")
    fig.colorbar(img_affichee, ax=ax, label='Valeur')

    ax_curseur_temps = plt.axes([0.20, 0.1, 0.65, 0.03])
    curseur_temps = Slider(
        ax=ax_curseur_temps,
        label='Pas de Temps',
        valmin=0,
        valmax=num_pas_temps_mat - 1,
        valinit=temps_initial,
        valstep=1 
    )

    def mettre_a_jour(val):
        temps_actuel = int(curseur_temps.val)
        img_affichee.set_data(donnees_matrice[:, :, temps_actuel])
        ax.set_title(f"Matrice ({num_lignes_mat}x{num_colonnes_mat}) au pas de temps : {temps_actuel}")
        fig.canvas.draw_idle()

    curseur_temps.on_changed(mettre_a_jour)
    ax.set_title(f"Matrice ({num_lignes_mat}x{num_colonnes_mat}) au pas de temps : {temps_initial}")
    plt.show()

if __name__ == "__main__":
    chemin_du_fichier = input("Veuillez entrer le chemin d'accès complet à votre fichier de données : ")

    matrice_3d = analyser_fichier_donnees_dynamique(chemin_du_fichier)

    if matrice_3d is not None:
        if matrice_3d.size > 0 :
            print(f"Affichage de la matrice de dimensions : {matrice_3d.shape} (Lignes, Colonnes, Temps)")
            afficher_matrice_interactive(matrice_3d)
        else:
            print("L'analyse a produit une matrice vide (taille 0). "
                  "Vérifiez les messages précédents et le fichier de données.")
    else:
        print("Impossible d'afficher la matrice car les données n'ont pas pu être chargées ou traitées.")