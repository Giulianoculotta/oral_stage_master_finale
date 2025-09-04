import matplotlib.pyplot as plt
import numpy as np

def charger_matrice_depuis_fichier_texte_v2(chemin_fichier):
    """
    Charge une matrice de données à partir d'un fichier texte spécifique.
    La fonction recherche un en-tête précis, puis parse les coordonnées de l'axe X
    et les données de la matrice qui suivent.
    """
    nom_en_tete_specifique = "______Output Data (col: scan axis, row: index axis)______"
    try:
        with open(chemin_fichier, 'r') as f:
            lignes = [ligne.strip() for ligne in f.readlines()]

        idx_en_tete_specifique = -1
        for i, ligne in enumerate(lignes):
            if nom_en_tete_specifique in ligne:
                idx_en_tete_specifique = i
                break

        if idx_en_tete_specifique == -1:
            print(f"Avertissement: En-tête non trouvé dans {chemin_fichier}.")
            return np.array([]), []

        idx_ligne_x_coords = idx_en_tete_specifique + 1
        if idx_ligne_x_coords >= len(lignes):
            return np.array([]), []

        ligne_x_coords = lignes[idx_ligne_x_coords]
        valeurs_ligne_x = ligne_x_coords.split(',')
        x_coords_parsed = []
        for val_str in valeurs_ligne_x[1:]:
            if val_str.strip() == '':
                continue
            try:
                x_coords_parsed.append(float(val_str))
            except ValueError:
                pass

        nombre_colonnes_attendu = len(x_coords_parsed)
        if nombre_colonnes_attendu == 0:
            return np.array([]), []

        idx_debut_donnees = idx_ligne_x_coords + 1
        if idx_debut_donnees >= len(lignes):
            return np.array([]), []

        matrice_extraite = []
        for ligne_brute in lignes[idx_debut_donnees:]:
            if not ligne_brute.strip():
                continue
            valeurs_str = ligne_brute.split(',')
            donnees_ligne_actuelle = []
            if len(valeurs_str) > 1:
                for val in valeurs_str[1:]:
                    try:
                        donnees_ligne_actuelle.append(float(val))
                    except ValueError:
                        donnees_ligne_actuelle.append(np.nan)
            else:
                continue
            
            if len(donnees_ligne_actuelle) < nombre_colonnes_attendu:
                donnees_ligne_actuelle.extend([np.nan] * (nombre_colonnes_attendu - len(donnees_ligne_actuelle)))
            elif len(donnees_ligne_actuelle) > nombre_colonnes_attendu:
                donnees_ligne_actuelle = donnees_ligne_actuelle[:nombre_colonnes_attendu]
            matrice_extraite.append(donnees_ligne_actuelle)

        if not matrice_extraite:
            return np.array([]), []

        return np.array(matrice_extraite), x_coords_parsed

    except FileNotFoundError:
        print(f"Erreur: Le fichier {chemin_fichier} est introuvable.")
        return np.array([]), []
    except Exception as e:
        print(f"Une erreur inattendue est survenue avec {chemin_fichier}: {e}")
        return np.array([]), []

def normaliser_par_reference(matrice, reference):
    """Normalise une matrice par une valeur de référence (min ou max)."""
    if matrice.size == 0 or reference is None or reference == 0:
        return np.copy(matrice) # Retourne une copie pour éviter les effets de bord
    return matrice.astype(float) / reference

def filtrer_matrice_par_frequence(matrice, x_coords, max_frequence):
    """Filtre les colonnes d'une matrice en fonction d'une fréquence maximale."""
    if matrice.size == 0 or not x_coords:
        return np.array([]), []
    indices_a_garder = [i for i, freq in enumerate(x_coords) if freq <= max_frequence]
    if not indices_a_garder:
        return np.array([]), []
    return matrice[:, indices_a_garder], [x_coords[i] for i in indices_a_garder]

def calculer_extremum_global(matrices, operation=np.nanmin):
    """
    Calcule un extremum global (min ou max) à partir d'une liste de matrices,
    en ignorant les zéros et les valeurs NaN.
    """
    all_data = []
    for m in matrices:
        if m.size > 0:
            all_data.append(m.flatten())
    if not all_data:
        return None
    combined_data = np.concatenate(all_data)
    filtered_data = combined_data[(combined_data != 0) & (~np.isnan(combined_data))]
    if filtered_data.size == 0:
        return None
    return operation(filtered_data)

def calculer_extremum_local(matrice, operation=np.nanmin):
    """Calcule l'extremum local d'une matrice, en ignorant les zéros et NaN."""
    if matrice.size == 0:
        return None
    filtered_data = matrice[(matrice != 0) & (~np.isnan(matrice))]
    if filtered_data.size == 0:
        return None
    return operation(filtered_data)

def afficher_comparaisons_min(nom_partie, matrices_orig, matrices_norm):
    """Affiche une grille 3x3 comparant les matrices originales et les normalisations par minimum."""
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    fig.suptitle(f'Analyses Comparatives (Normalisation par Minimum) - {nom_partie}', fontsize=20)

    types_test = ['avant', 'post1', 'post2']
    labels_lignes = ['Originale', 'Norm. Min Global', 'Norm. Min Local']

    for i, label in enumerate(labels_lignes):
        axes[i, 0].set_ylabel(label, fontsize=14, labelpad=20)

    for col, test in enumerate(types_test):
        # Ligne 0: Originales
        mat = matrices_orig[test]
        ax = axes[0, col]
        ax.set_title(f'{test.capitalize()} Test (Originale)', fontsize=12)
        if mat.size > 0:
            im = ax.imshow(mat, aspect='auto', cmap='viridis')
            fig.colorbar(im, ax=ax)
        else:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)

        # Ligne 1: Normalisation Min Global
        mat_gmin = matrices_norm[test]['global_min']
        ax_gmin = axes[1, col]
        ax_gmin.set_title(f'{test.capitalize()} - {labels_lignes[1]}', fontsize=12)
        if mat_gmin.size > 0:
            im = ax_gmin.imshow(mat_gmin, aspect='auto', cmap='viridis')
            fig.colorbar(im, ax=ax_gmin)
        else:
            ax_gmin.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax_gmin.transAxes)

        # Ligne 2: Normalisation Min Local
        mat_lmin = matrices_norm[test]['local_min']
        ax_lmin = axes[2, col]
        ax_lmin.set_title(f'{test.capitalize()} - {labels_lignes[2]}', fontsize=12)
        if mat_lmin.size > 0:
            im = ax_lmin.imshow(mat_lmin, aspect='auto', cmap='viridis')
            fig.colorbar(im, ax=ax_lmin)
        else:
            ax_lmin.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax_lmin.transAxes)

    plt.tight_layout(rect=[0.03, 0, 1, 0.96])
    plt.show()

def afficher_comparaisons_max(nom_partie, matrices_orig, matrices_norm):
    """Affiche une grille 3x3 comparant les matrices originales et les normalisations par maximum."""
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    fig.suptitle(f'Analyses Comparatives (Normalisation par Maximum) - {nom_partie}', fontsize=20)

    types_test = ['avant', 'post1', 'post2']
    labels_lignes = ['Originale', 'Norm. Max Global', 'Norm. Max Local']

    for i, label in enumerate(labels_lignes):
        axes[i, 0].set_ylabel(label, fontsize=14, labelpad=20)

    for col, test in enumerate(types_test):
        # Ligne 0: Originales
        mat = matrices_orig[test]
        ax = axes[0, col]
        ax.set_title(f'{test.capitalize()} Test (Originale)', fontsize=12)
        if mat.size > 0:
            im = ax.imshow(mat, aspect='auto', cmap='viridis')
            fig.colorbar(im, ax=ax)
        else:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)

        # Ligne 1: Normalisation Max Global
        mat_gmax = matrices_norm[test]['global_max']
        ax_gmax = axes[1, col]
        ax_gmax.set_title(f'{test.capitalize()} - {labels_lignes[1]}', fontsize=12)
        if mat_gmax.size > 0:
            im = ax_gmax.imshow(mat_gmax, aspect='auto', cmap='viridis')
            fig.colorbar(im, ax=ax_gmax)
        else:
            ax_gmax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax_gmax.transAxes)

        # Ligne 2: Normalisation Max Local
        mat_lmax = matrices_norm[test]['local_max']
        ax_lmax = axes[2, col]
        ax_lmax.set_title(f'{test.capitalize()} - {labels_lignes[2]}', fontsize=12)
        if mat_lmax.size > 0:
            im = ax_lmax.imshow(mat_lmax, aspect='auto', cmap='viridis')
            fig.colorbar(im, ax=ax_lmax)
        else:
            ax_lmax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax_lmax.transAxes)

    plt.tight_layout(rect=[0.03, 0, 1, 0.96])
    plt.show()



# 1. Définir les chemins des fichiers
chemins = {
    'p1': {"avant": "MB1-pre.csv", "post1": "MB1-post.csv", "post2": "MB1-post2bis.csv"},
    'p2': {"avant": "MB2-pre.csv", "post1": "MB2-post.csv", "post2": "MB2-post2bis.csv"},
    'p3': {"avant": "MB3-pre.csv", "post1": "MB3-post.csv", "post2": "MB3-post2bis.csv"}
}

# 2. Charger toutes les matrices brutes
matrices_brutes, x_coords_brutes = {}, {}
for p in ['p1', 'p2', 'p3']:
    matrices_brutes[p], x_coords_brutes[p] = {}, {}
    for test, chemin in chemins[p].items():
        matrices_brutes[p][test], x_coords_brutes[p][test] = charger_matrice_depuis_fichier_texte_v2(chemin)

# 3. Appliquer les transformations pour obtenir les matrices "originales" à afficher
matrices_orig = {}
# Partie 1
p1_avant = matrices_brutes['p1']['avant']
p1_post1_temp = matrices_brutes['p1']['post1']
p1_post1 = np.flip(np.rot90(p1_post1_temp)) if p1_post1_temp.size > 0 else np.array([])
p1_post2_raw = np.flip(matrices_brutes['p1']['post2'])
p1_post2, _ = filtrer_matrice_par_frequence(p1_post2_raw, x_coords_brutes['p1']['post2'], 215)
if p1_post2.size > 0: p1_post2[p1_post2 > 10] = np.nan
matrices_orig['p1'] = {'avant': p1_avant, 'post1': p1_post1, 'post2': p1_post2}
# Partie 2
p2_avant = matrices_brutes['p2']['avant']
p2_post1_temp = matrices_brutes['p2']['post1']
p2_post1 = np.rot90(p2_post1_temp) if p2_post1_temp.size > 0 else np.array([])
matrices_orig['p2'] = {'avant': p2_avant, 'post1': p2_post1, 'post2': matrices_brutes['p2']['post2']}
# Partie 3
p3_avant = matrices_brutes['p3']['avant']
p3_post1_temp = matrices_brutes['p3']['post1']
p3_post1 = np.flip(np.rot90(p3_post1_temp), axis=1) if p3_post1_temp.size > 0 else np.array([])
p3_post2_temp = matrices_brutes['p3']['post2']
p3_post2 = np.flip(p3_post2_temp, axis=1) if p3_post2_temp.size > 0 else np.array([])
matrices_orig['p3'] = {'avant': p3_avant, 'post1': p3_post1, 'post2': p3_post2}

# 4. Calculer tous les extrema (globaux et locaux)
extrema = {'global': {}, 'local': {p: {} for p in ['p1', 'p2', 'p3']}}
for test in ['avant', 'post1', 'post2']:
    # Extrema Globaux
    mats_globales = [matrices_orig[p][test] for p in ['p1', 'p2', 'p3']]
    extrema['global'][f'{test}_min'] = calculer_extremum_global(mats_globales, np.nanmin)
    extrema['global'][f'{test}_max'] = calculer_extremum_global(mats_globales, np.nanmax)
    # Extrema Locaux
    for p in ['p1', 'p2', 'p3']:
        mat = matrices_orig[p][test]
        extrema['local'][p][f'{test}_min'] = calculer_extremum_local(mat, np.nanmin)
        extrema['local'][p][f'{test}_max'] = calculer_extremum_local(mat, np.nanmax)

# 5. Effectuer toutes les normalisations
matrices_norm = {p: {t: {} for t in ['avant', 'post1', 'post2']} for p in ['p1', 'p2', 'p3']}
for p in ['p1', 'p2', 'p3']:
    for test in ['avant', 'post1', 'post2']:
        mat_orig = matrices_orig[p][test]
        matrices_norm[p][test]['global_min'] = normaliser_par_reference(mat_orig, extrema['global'][f'{test}_min'])
        matrices_norm[p][test]['global_max'] = normaliser_par_reference(mat_orig, extrema['global'][f'{test}_max'])
        matrices_norm[p][test]['local_min'] = normaliser_par_reference(mat_orig, extrema['local'][p][f'{test}_min'])
        matrices_norm[p][test]['local_max'] = normaliser_par_reference(mat_orig, extrema['local'][p][f'{test}_max'])

# 6. Afficher les résultats complets pour chaque partie
for i in range(1, 4):
    partie_key = f'p{i}'
    nom_partie_affichage = f'Partie {i}'
    # Afficher la figure pour les normalisations par minimum
    afficher_comparaisons_min(
        nom_partie_affichage,
        matrices_orig[partie_key],
        matrices_norm[partie_key]
    )
    # Afficher la figure pour les normalisations par maximum
    afficher_comparaisons_max(
        nom_partie_affichage,
        matrices_orig[partie_key],
        matrices_norm[partie_key]
    )
