import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Constante pour identifier le début des données, basée sur l'image.
LIGNE_DEBUT_DONNEES = "______Output Data (Format for each row: index Number, scanning number, Waveform)______"
# Nombre de pas de temps (longueur de la forme d'onde).
# Basé sur "Waveform Length (pts): 811" dans l'image.
# Si cela doit aussi être dynamique, il faudrait l'adapter.
NOMBRE_PAS_TEMPS = 811

def analyser_fichier_donnees_dynamique(chemin_fichier):
    """
    Analyse le fichier de données pour extraire la matrice 3D,
    en déterminant dynamiquement le nombre de lignes et de colonnes.
    """
    donnees_brutes = []  # Pour stocker (idx_ligne, idx_colonne, waveform_array)
    max_idx_ligne = -1
    max_idx_colonne = -1
    
    en_tete_trouve = False
    lignes_donnees_valides_lues = 0
    
    print(f"Analyse dynamique du fichier : {chemin_fichier}")
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

                if not ligne_nettoyee:  # Ignorer les lignes vides après l'en-tête
                    continue

                elements = ligne_nettoyee.split()
                
                # Chaque ligne de données doit avoir au moins idx_ligne, idx_colonne et une valeur de forme d'onde.
                # La longueur exacte de la forme d'onde est vérifiée par rapport à NOMBRE_PAS_TEMPS.
                if len(elements) != 2 + NOMBRE_PAS_TEMPS:
                    if en_tete_trouve: # N'afficher les avertissements qu'après avoir trouvé l'en-tête
                        print(f"Info (Ligne {num_ligne_fichier} - Phase 1): La ligne ne correspond pas au format attendu "
                              f"(2 indices + {NOMBRE_PAS_TEMPS} valeurs de forme d'onde). "
                              f"Nombre d'éléments trouvés: {len(elements)}. Contenu : '{ligne_nettoyee[:70]}...'. Ligne ignorée.")
                    continue

                try:
                    idx_ligne = int(elements[0])
                    idx_colonne = int(elements[1])
                    
                    # Les éléments restants forment la forme d'onde
                    waveform_str_values = elements[2:]
                    # Vérification explicite (redondante si len(elements) est déjà vérifié, mais plus sûr)
                    if len(waveform_str_values) == NOMBRE_PAS_TEMPS:
                        forme_onde = np.array([float(x) for x in waveform_str_values])
                        
                        donnees_brutes.append((idx_ligne, idx_colonne, forme_onde))
                        max_idx_ligne = max(max_idx_ligne, idx_ligne)
                        max_idx_colonne = max(max_idx_colonne, idx_colonne)
                        lignes_donnees_valides_lues += 1
                    else:
                        # Ce cas ne devrait pas être atteint si la vérification len(elements) est correcte
                        print(f"Avertissement (Ligne {num_ligne_fichier} - Phase 1): Longueur de forme d'onde ({len(waveform_str_values)}) "
                              f"ne correspond pas à NOMBRE_PAS_TEMPS ({NOMBRE_PAS_TEMPS}). Ligne ignorée.")

                except ValueError:
                    if en_tete_trouve:
                        print(f"Avertissement (Ligne {num_ligne_fichier} - Phase 1): Impossible de convertir les indices ou "
                              f"valeurs de forme d'onde en nombres : '{ligne_nettoyee[:70]}...'. Ligne ignorée.")
                # IndexError ne devrait pas se produire grâce aux vérifications de longueur précédentes.
    
    except FileNotFoundError:
        print(f"Erreur : Le fichier '{chemin_fichier}' n'a pas été trouvé.")
        return None
    except Exception as e:
        print(f"Une erreur inattendue est survenue lors de la lecture du fichier (Phase 1) : {e}")
        return None

    if not en_tete_trouve:
        print(f"L'en-tête '{LIGNE_DEBUT_DONNEES}' n'a pas été trouvé. Impossible de traiter le fichier.")
        return None
        
    if not donnees_brutes:
        print("Aucune donnée valide n'a été lue après l'en-tête.")
        return None

    # +1 car les indices commencent à 0
    nombre_lignes_detectees = max_idx_ligne + 1
    nombre_colonnes_detectees = max_idx_colonne + 1
    
    print(f"Fin de la Phase 1 : {lignes_donnees_valides_lues} lignes de données valides lues.")
    print(f"Dimensions détectées pour la matrice : {nombre_lignes_detectees} lignes (max indice lu: {max_idx_ligne}), "
          f"{nombre_colonnes_detectees} colonnes (max indice lu: {max_idx_colonne}).")
    print(f"Nombre de pas de temps utilisé (longueur de forme d'onde) : {NOMBRE_PAS_TEMPS}.")

    # Phase 2: Création et remplissage de la matrice NumPy
    print("Phase 2 : Création de la matrice NumPy et remplissage des données...")
    try:
        if nombre_lignes_detectees <= 0 or nombre_colonnes_detectees <= 0:
            print(f"Erreur : Dimensions détectées non valides ({nombre_lignes_detectees}x{nombre_colonnes_detectees}). "
                  "Impossible de créer la matrice.")
            return None

        donnees_matrice = np.zeros((nombre_lignes_detectees, nombre_colonnes_detectees, NOMBRE_PAS_TEMPS))
        
        lignes_remplies = 0
        for idx_l, idx_c, forme_onde_data in donnees_brutes:
            # Vérification supplémentaire (devrait toujours être vrai si la logique est correcte)
            if 0 <= idx_l < nombre_lignes_detectees and 0 <= idx_c < nombre_colonnes_detectees:
                donnees_matrice[idx_l, idx_c, :] = forme_onde_data
                lignes_remplies +=1
            else:
                # Ceci indiquerait une erreur dans la logique de détection des max_idx ou de stockage.
                print(f"Erreur interne critique (Phase 2): Indice ({idx_l}, {idx_c}) hors des limites détectées "
                      f"({nombre_lignes_detectees}x{nombre_colonnes_detectees}) lors du remplissage. Donnée ignorée.")
        
        print(f"Matrice NumPy remplie. {lignes_remplies} points (ligne,colonne) ont été renseignés dans la matrice.")
        if lignes_remplies != lignes_donnees_valides_lues:
             print(f"Avertissement: {lignes_donnees_valides_lues} lignes de données valides ont été lues, "
                   f"mais seulement {lignes_remplies} ont été insérées dans la matrice finale. "
                   "Vérifiez les messages d'erreur précédents.")
        return donnees_matrice
        
    except ValueError as ve: 
        print(f"Erreur lors de la création de la matrice NumPy (Phase 2) : {ve}")
        return None
    except Exception as e:
        print(f"Une erreur inattendue est survenue lors de la création ou du remplissage de la matrice (Phase 2) : {e}")
        return None


def afficher_matrice_interactive(donnees_matrice):
    """
    Affiche la matrice interactivement avec un curseur pour le temps.
    S'adapte aux dimensions de la matrice fournie.
    """
    if donnees_matrice is None or donnees_matrice.size == 0:
        print("Aucune donnée à afficher ou matrice vide.")
        return

    # Obtient les dimensions directement depuis la matrice fournie
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
    plt.subplots_adjust(bottom=0.25) # Espace pour le curseur

    temps_initial = 0
    # `origin='lower'` place (0,0) en bas à gauche. `aspect='auto'` ajuste les proportions.
    img_affichee = ax.imshow(donnees_matrice[:, :, temps_initial], cmap='viridis', aspect='auto', origin='lower')
    
    ax.set_xlabel(f"Indice de Colonne (0 à {num_colonnes_mat - 1})")
    ax.set_ylabel(f"Indice de Ligne (0 à {num_lignes_mat - 1})")
    fig.colorbar(img_affichee, ax=ax, label='Valeur')

    # Création du curseur pour naviguer dans le temps
    ax_curseur_temps = plt.axes([0.20, 0.1, 0.65, 0.03]) # Position [gauche, bas, largeur, hauteur]
    curseur_temps = Slider(
        ax=ax_curseur_temps,
        label='Pas de Temps',
        valmin=0,
        valmax=num_pas_temps_mat - 1, # Utilise le nombre de pas de temps de la matrice
        valinit=temps_initial,
        valstep=1 
    )

    # Fonction appelée quand la valeur du curseur change
    def mettre_a_jour(val):
        temps_actuel = int(curseur_temps.val)
        img_affichee.set_data(donnees_matrice[:, :, temps_actuel])
        # Optionnel: Mettre à jour les limites de couleur pour chaque slice si désiré
        # vmin, vmax = np.min(donnees_matrice[:, :, temps_actuel]), np.max(donnees_matrice[:, :, temps_actuel])
        # if vmin < vmax: # Évite l'erreur si toutes les valeurs sont identiques
        #    img_affichee.set_clim(vmin=vmin, vmax=vmax)
        # else: # Gère le cas où toutes les valeurs sont identiques
        #    img_affichee.set_clim(vmin=vmin - 0.5, vmax=vmax + 0.5) # ou une autre logique appropriée

        ax.set_title(f"Matrice ({num_lignes_mat}x{num_colonnes_mat}) au pas de temps : {temps_actuel}")
        fig.canvas.draw_idle() # Redessine la figure

    curseur_temps.on_changed(mettre_a_jour)

    # Titre initial
    ax.set_title(f"Matrice ({num_lignes_mat}x{num_colonnes_mat}) au pas de temps : {temps_initial}")
    plt.show()

if __name__ == "__main__":
    chemin_du_fichier = input("Veuillez entrer le chemin d'accès complet à votre fichier de données : ")

    # Analyse les données en déterminant dynamiquement les dimensions
    matrice_3d = analyser_fichier_donnees_dynamique(chemin_du_fichier)

    if matrice_3d is not None:
        if matrice_3d.size > 0 : # Vérifie si la matrice contient des éléments
            print(f"Affichage de la matrice de dimensions : {matrice_3d.shape}")
            afficher_matrice_interactive(matrice_3d)
        else: # matrice_3d n'est pas None mais est vide (ex: np.zeros((0,0,811)))
            print("L'analyse a produit une matrice vide (taille 0). Vérifiez les messages précédents et le fichier de données.")
    else:
        print("Impossible d'afficher la matrice car les données n'ont pas pu être chargées ou traitées correctement.")