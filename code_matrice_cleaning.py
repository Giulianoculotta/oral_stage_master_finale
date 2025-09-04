
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import os
import sys
import traceback 

# --- Constantes et Configuration ---

script_dir = os.path.dirname(os.path.abspath(__file__))

FICHIER_DATA = os.path.join(script_dir, 'C:/Users/sword/Desktop/Données US/Pre╠ü csan-CH1.csv')
print(f"Chemin fichier de données utilisé: {FICHIER_DATA}")

CRITERE_BORDURE = 0.95
DELAI_ANIMATION_MS = 150 # Speed for the "film" playback (milliseconds)
HEADER_END_MARKER = "______Output Data (Format for each row: index Number, scanning number, Waveform)______"
EXPECTED_SCANS_PER_MOMENT = 70 

# --- Fonctions de Nettoyage ---
def imputer_nan_par_moyenne_voisins(matrix):

    if matrix.size == 0: return matrix
    rows, cols = matrix.shape
    matrice_imputee = matrix.copy()
    nan_coords = np.argwhere(np.isnan(matrix))
    for r, c in nan_coords:
        voisins_valides = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0: continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    valeur_voisin = matrix[nr, nc]
                    if not np.isnan(valeur_voisin): voisins_valides.append(valeur_voisin)
        if voisins_valides: matrice_imputee[r, c] = np.mean(voisins_valides)
        else: matrice_imputee[r, c] = 0 # Ou np.nan
    return matrice_imputee

def nettoyer_matrice(matrix_brute):

    if matrix_brute.size == 0: return matrix_brute
    mat = matrix_brute.copy()
    while True:
        bord_supprime = False
        rows, cols = mat.shape
        if rows == 0 or cols == 0: break
        # Check Haut border
        if rows > 0:
            valid_count = np.count_nonzero(~np.isnan(mat[0, :])); valid_ratio = valid_count/cols if cols > 0 else 0
            if valid_ratio < CRITERE_BORDURE: mat = mat[1:, :]; bord_supprime = True; rows -= 1
            if rows == 0: break
        # Check Bas border
        if rows > 0:
             valid_count = np.count_nonzero(~np.isnan(mat[-1, :])); valid_ratio = valid_count/cols if cols > 0 else 0
             if valid_ratio < CRITERE_BORDURE: mat = mat[:-1, :]; bord_supprime = True; rows -= 1
             if rows == 0: break
        # Check Gauche border
        if cols > 0:
            valid_count = np.count_nonzero(~np.isnan(mat[:, 0])); valid_ratio = valid_count/rows if rows > 0 else 0
            if valid_ratio < CRITERE_BORDURE: mat = mat[:, 1:]; bord_supprime = True; cols -= 1
            if cols == 0: break
        # Check Droite border
        if cols > 0:
            valid_count = np.count_nonzero(~np.isnan(mat[:, -1])); valid_ratio = valid_count/rows if rows > 0 else 0
            if valid_ratio < CRITERE_BORDURE: mat = mat[:, :-1]; bord_supprime = True; cols -= 1
            if cols == 0: break
        # Stop if no border was removed
        if not bord_supprime: break
    matrice_elaguee = mat
    matrice_finale = imputer_nan_par_moyenne_voisins(matrice_elaguee)
    return matrice_finale


# --- Helper function pour traiter un bloc (moment, scan) ---
def _process_matrix_block(waveforms, matrix_index_pair):
    """Assemble, clean, and return a single matrix block."""
    if not waveforms: return None
    index_str = f"({matrix_index_pair[0]}, {matrix_index_pair[1]})"
    try:
        max_len = max(len(wf) for wf in waveforms) # Max length within this specific block
        # print(f"    Processing block for Index Pair {index_str}: {len(waveforms)} lignes, max length {max_len}") # Verbose
        mat_brute = np.full((len(waveforms), max_len), np.nan, dtype=float)
        for i, wf in enumerate(waveforms):
            mat_brute[i, :len(wf)] = wf
        mat_nettoyee = nettoyer_matrice(mat_brute)
        if mat_nettoyee.size > 0:
            return mat_nettoyee
        else: return None # Return None if cleaning made it empty
    except ValueError: # Catch cases like empty waveforms list after filtering?
         print(f"  ERREUR (ValueError) pendant le traitement du bloc pour Index Pair {index_str}. Waveforms count: {len(waveforms)}", file=sys.stderr)
         return None
    except Exception as e:
        print(f"  ERREUR pendant le traitement du bloc pour Index Pair {index_str}: {e}", file=sys.stderr)
        traceback.print_exc()
        return None

# --- Chargement des Données (Groupé par Moment Index, puis Scan Index) ---
def charger_matrices_par_moment(filepath, header_marker=HEADER_END_MARKER):
    """
    Charge les données groupées par le premier index (Moment Index).
    Pour chaque Moment Index, retourne une liste de matrices ordonnée
    par le second index (Scan Index).
    Retourne un dictionnaire: {MomentIndex: [MatrixScan0, MatrixScan1, ...]}
    """
    print(f"Chargement et groupage par Moment Index depuis '{filepath}'...")
    # Structure intermédiaire: {moment_idx: {scan_idx: [waveform_list_1,...]}}
    data_by_moment_scan = {}
    found_data_start = False
    line_num = 0

    # --- Phase 1: Lire toutes les lignes et grouper les waveforms ---
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            # Chercher le marqueur
            for line in f:
                line_num += 1
                if header_marker in line:
                    found_data_start = True; break
            if not found_data_start:
                print(f"ERREUR: Marqueur '{header_marker}' non trouvé.", file=sys.stderr); return {}

            # Lire les lignes de données
            print("Marqueur trouvé. Lecture des lignes de données...")
            data_line_count = 0
            error_count = 0
            for line in f:
                line_num += 1
                stripped_line = line.strip()
                if not stripped_line: continue # Ignorer lignes vides
                parts = stripped_line.split()
                if len(parts) < 3: continue # Ignorer lignes trop courtes

                # Extraire Moment Index, Scan Index, et Waveform
                try:
                    moment_idx = int(parts[0])
                    scan_idx = int(parts[1])
                    waveform_float_values = [float(val) for val in parts[2:]]
                except ValueError:
                    error_count += 1
                    if error_count < 20: # Limit error printing
                       print(f"Alerte Ligne {line_num}: Données non-numériques, ignorée.", file=sys.stderr)
                    elif error_count == 20:
                       print("Alerte: Trop d'erreurs de conversion numérique, messages suivants supprimés.", file=sys.stderr)
                    continue

                # Stocker dans la structure intermédiaire
                if moment_idx not in data_by_moment_scan:
                    data_by_moment_scan[moment_idx] = {}
                if scan_idx not in data_by_moment_scan[moment_idx]:
                    data_by_moment_scan[moment_idx][scan_idx] = []
                # Ajouter *toutes* les lignes correspondant à ce moment/scan
                data_by_moment_scan[moment_idx][scan_idx].append(waveform_float_values)
                data_line_count += 1

            print(f"Lecture terminée. {data_line_count} lignes de données lues. {error_count} erreurs de conversion.")

    except FileNotFoundError: print(f"ERREUR: Fichier '{filepath}' non trouvé.", file=sys.stderr); return {}
    except Exception as e: print(f"ERREUR lecture fichier: {e}", file=sys.stderr); traceback.print_exc(); return {}

    if not data_by_moment_scan:
        print("Aucune donnée valide trouvée après le marqueur.", file=sys.stderr); return {}

    # --- Phase 2: Convertir les groupes de waveforms en matrices nettoyées ---
    final_grouped_matrices = {}
    print("Traitement des groupes de waveforms en matrices...")
    sorted_moment_indices = sorted(data_by_moment_scan.keys())

    total_matrices_processed = 0
    total_matrices_failed = 0

    for moment_idx in sorted_moment_indices:
        # print(f"Traitement Moment Index {moment_idx}...") # Verbose
        matrices_for_this_moment_dict = {} # Use dict for sorting by scan_idx later
        scan_data_for_moment = data_by_moment_scan[moment_idx]
        sorted_scan_indices = sorted(scan_data_for_moment.keys()) # Should be 0-69 ideally

        num_scans_found = len(sorted_scan_indices)
        if num_scans_found != EXPECTED_SCANS_PER_MOMENT:
             print(f"  Alerte: Moment Index {moment_idx} a {num_scans_found} scans au lieu des {EXPECTED_SCANS_PER_MOMENT} attendus.", file=sys.stderr)

        processed_count_this_moment = 0
        for scan_idx in sorted_scan_indices:
            # On s'attend à ce que scan_idx soit déjà le bon index (0-69)
            waveforms_for_scan = scan_data_for_moment[scan_idx]
            # Chaque paire (moment, scan) forme maintenant une matrice
            processed_matrix = _process_matrix_block(waveforms_for_scan, (moment_idx, scan_idx))
            if processed_matrix is not None:
                matrices_for_this_moment_dict[scan_idx] = processed_matrix
                processed_count_this_moment += 1
                total_matrices_processed += 1
            else:
                 total_matrices_failed += 1

        matrices_list = []
        found_valid_matrix = False
        for i in range(EXPECTED_SCANS_PER_MOMENT): # Iterate 0 to 69
            matrix = matrices_for_this_moment_dict.get(i) # Get matrix if processed
            if matrix is not None:
                matrices_list.append(matrix)
                found_valid_matrix = True
            else:

                matrices_list.append(np.array([[np.nan]])) 
                if i in sorted_scan_indices: 
                     print(f"  Alerte: Placeholder ajouté pour échec traitement Moment={moment_idx}, Scan={i}.", file=sys.stderr)



        if found_valid_matrix: 
            final_grouped_matrices[moment_idx] = matrices_list

        else:
            print(f"  Alerte: Aucune matrice valide générée pour Moment Index {moment_idx}.", file=sys.stderr)


    print(f"Traitement terminé. {total_matrices_processed} matrices traitées avec succès, {total_matrices_failed} échecs.")
    print(f"{len(final_grouped_matrices)} Moment Indices avec au moins une matrice valide.")
    return final_grouped_matrices


# --- Classe de Visualisation (par Moment et Scan - Version Précédente) ---
class VisualiseurMoments:
    def __init__(self, grouped_matrices_dict):
        if not grouped_matrices_dict:
            print("ERREUR: Aucune donnée fournie au visualiseur.", file=sys.stderr)
            self._show_error_figure("Aucune donnée chargée")
            return

        self.grouped_matrices = grouped_matrices_dict
        self.moment_indices = sorted(self.grouped_matrices.keys())
        self.num_moments = len(self.moment_indices)

        if self.num_moments == 0:
             print("ERREUR: Dictionnaire de données vide après filtrage.", file=sys.stderr)
             self._show_error_figure("Données vides après filtrage")
             return

        # Initialisation des index
        self.current_moment_slider_idx = 0
        self.update_current_moment_data() 
        self.current_scan_idx = 0

        # Initialisation animation
        self.animation_timer = None
        self.is_playing = False

        # Calculer vmin/vmax globalement
        self.calculate_global_vmin_vmax()
        if self.vmin is None: 
             self._show_error_figure("Impossible de calculer vmin/vmax")
             return

        # --- Initialisation de l'interface graphique ---
        self.fig, self.ax = plt.subplots(figsize=(9, 8))
        plt.subplots_adjust(left=0.1, bottom=0.3, right=0.9, top=0.9)

        # Afficher la première matrice valide du premier moment
        mat_init, self.current_scan_idx = self._get_valid_matrix(self.current_scan_idx)
        self.image = self.ax.imshow(mat_init, cmap='viridis', vmin=self.vmin, vmax=self.vmax, origin='upper', interpolation='nearest')
        self.cbar = self.fig.colorbar(self.image, ax=self.ax)
        self.cbar.set_label('Valeurs')
        self.update_title()


        axcolor = 'lightgoldenrodyellow'

        # Slider 1: Sélection du Moment Index
        self.ax_slider_moment = plt.axes([0.15, 0.15, 0.7, 0.03], facecolor=axcolor)
        moment_label = f'Moment Index ({self.moment_indices[0]}..{self.moment_indices[-1]})' if self.num_moments > 0 else 'Moment Index'
        self.slider_moment = Slider(
            ax=self.ax_slider_moment, label=moment_label,
            valmin=0, valmax=max(0, self.num_moments - 1),
            valinit=self.current_moment_slider_idx, valstep=1
        )
        self.slider_moment.on_changed(self.on_moment_change)

        # Slider 2: Sélection du Scan Index (frame du film)
        self.ax_slider_scan = plt.axes([0.15, 0.08, 0.7, 0.03], facecolor=axcolor)
        scan_slider_max = max(0, self.num_scans_current_moment - 1)
        self.slider_scan = Slider(
            ax=self.ax_slider_scan, label='Scan Index (Frame 0-69)',
            valmin=0, valmax=scan_slider_max,
            valinit=self.current_scan_idx, valstep=1
        )
        self.slider_scan.on_changed(self.on_scan_change)

        # Boutons de contrôle pour le "film" (Scan Index)
        self.ax_prev = plt.axes([0.25, 0.02, 0.1, 0.04])
        self.btn_prev = Button(self.ax_prev, '< Scan')
        self.btn_prev.on_clicked(self.prev_scan)

        self.ax_play = plt.axes([0.40, 0.02, 0.1, 0.04])
        self.btn_play = Button(self.ax_play, '▶ Play')
        self.btn_play.on_clicked(self.play_pause)

        self.ax_next = plt.axes([0.55, 0.02, 0.1, 0.04])
        self.btn_next = Button(self.ax_next, 'Scan >')
        self.btn_next.on_clicked(self.next_scan)

        # Activer/Désactiver contrôles de scan initiaux
        self.update_scan_controls_state()
        plt.show()

    def _show_error_figure(self, message):
        """Displays a simple plot with an error message."""
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, message, ha='center', va='center', color='red', fontsize=12)
        ax.axis('off')
        plt.show()

    def _get_valid_matrix(self, start_scan_idx):
        """Trouve la première matrice valide à partir de start_scan_idx, ou une matrice vide."""
        if self.num_scans_current_moment == 0:
            return np.array([[np.nan]]), 0 # Retourne placeholder si aucune matrice

        # Cherche une matrice non-placeholder à partir de l'index de scan actuel
        for i in range(self.num_scans_current_moment):
            check_idx = (start_scan_idx + i) % self.num_scans_current_moment
            matrix = self.current_moment_matrices[check_idx]

            if matrix.size > 0 and not (matrix.shape == (1, 1) and np.isnan(matrix[0, 0])):
                return matrix, check_idx # Retourne la matrice valide et son index


        return np.array([[np.nan]]), start_scan_idx # Retourne placeholder

    def update_current_moment_data(self):
        """Met à jour les données relatives au moment sélectionné par le slider."""
        self.current_moment_actual_idx = self.moment_indices[self.current_moment_slider_idx]
        self.current_moment_matrices = self.grouped_matrices.get(self.current_moment_actual_idx, []) # Utiliser .get pour sécurité
        self.num_scans_current_moment = len(self.current_moment_matrices)
        # print(f"Moment {self.current_moment_actual_idx} selected, {self.num_scans_current_moment} scans found.")


    def calculate_global_vmin_vmax(self):
        """Calcule vmin/vmax sur toutes les matrices valides."""
        print("Calcul de vmin/vmax global...")
        all_valid_data_list = []
        valid_matrix_found = False
        for moment_idx in self.moment_indices:
            for matrix in self.grouped_matrices.get(moment_idx, []): # Utiliser .get
                 # Vérifier si ce n'est pas un placeholder et contient des données valides
                 if matrix is not None and matrix.size > 0 and not (matrix.shape == (1,1) and np.isnan(matrix[0,0])):
                     valid_non_nan = matrix[~np.isnan(matrix)]
                     if valid_non_nan.size > 0:
                          all_valid_data_list.append(valid_non_nan.flatten())
                          valid_matrix_found = True

        if not valid_matrix_found:
            print("Alerte: Aucune donnée numérique valide trouvée. vmin/vmax non définis.", file=sys.stderr)
            self.vmin, self.vmax = None, None # Indiquer l'échec
            return

        try:
            all_valid_data = np.concatenate(all_valid_data_list)
            if all_valid_data.size == 0: raise ValueError("Données concaténées vides.")
            self.vmin = np.nanmin(all_valid_data)
            self.vmax = np.nanmax(all_valid_data)
            print(f"vmin global={self.vmin}, vmax global={self.vmax}")
        except Exception as e:
            print(f"Erreur calcul vmin/vmax ({e}), non définis.", file=sys.stderr)
            self.vmin, self.vmax = None, None # Indiquer l'échec
            return

        # Assurer vmin != vmax si le calcul a réussi
        if self.vmin is not None and self.vmin == self.vmax:
             offset = max(abs(self.vmin) * 1e-6, 1e-6)
             self.vmax += offset
             if self.vmin >= self.vmax: self.vmin -= offset

    def update_title(self):
        """Met à jour le titre du graphique."""
        title = f"Moment Index: {self.current_moment_actual_idx} | Scan (Frame): {self.current_scan_idx + 1}/{self.num_scans_current_moment}"
        self.ax.set_title(title)

    def update_scan_controls_state(self):
        """Active/désactive les contrôles de scan basé sur le nb de scans."""
        enabled = self.num_scans_current_moment > 1
        self.slider_scan.ax.set_visible(enabled)
        # self.ax_slider_scan.set_visible(enabled) # Ne pas cacher l'axe entier?
        self.btn_prev.ax.set_visible(enabled)
        self.btn_play.ax.set_visible(enabled)
        self.btn_next.ax.set_visible(enabled)
        if not enabled and self.is_playing: self.play_pause()

    def on_moment_change(self, val):
        """Appelé lorsque le slider Moment Index change."""
        new_moment_slider_idx = int(round(val))
        if new_moment_slider_idx == self.current_moment_slider_idx: return

        # Mettre à jour l'index du moment et les données associées
        self.current_moment_slider_idx = new_moment_slider_idx
        self.update_current_moment_data()

        # Réinitialiser le scan index et le slider de scan
        self.current_scan_idx = 0 # Reset scan index to 0 for the new moment
        scan_slider_max = max(0, self.num_scans_current_moment - 1)

        self.slider_scan.valmax = scan_slider_max
        # S'assurer que xlim couvre au moins 0-1 pour éviter les erreurs si max est 0
        self.slider_scan.ax.set_xlim(0, scan_slider_max if scan_slider_max > 0 else 1)

        self.slider_scan.eventson = False
        # Réinitialiser la valeur du slider de scan à 0 (ou au premier index valide?)
        _, self.current_scan_idx = self._get_valid_matrix(0) # Trouver le premier scan valide
        self.slider_scan.set_val(self.current_scan_idx)
        self.slider_scan.eventson = True

        self.update_scan_controls_state()
        self.update_display() # Afficher la première (ou première valide) matrice du nouveau moment

    def on_scan_change(self, val):
        """Appelé lorsque le slider Scan Index change."""
        new_scan_idx = int(round(val))
        if new_scan_idx == self.current_scan_idx: return
        if 0 <= new_scan_idx < self.num_scans_current_moment:
            self.current_scan_idx = new_scan_idx
            self.update_display()
        else:
            self.slider_scan.eventson = False
            self.slider_scan.set_val(self.current_scan_idx) # Reset to current if out of bounds
            self.slider_scan.eventson = True


    def update_display(self):
        """Met à jour l'image affichée."""
        if not (0 <= self.current_scan_idx < self.num_scans_current_moment):
             matrix = np.array([[np.nan]]) # Placeholder si index invalide
        else:
             matrix = self.current_moment_matrices[self.current_scan_idx]
             # Check if it's the placeholder we added during loading
             if matrix.shape == (1,1) and np.isnan(matrix[0,0]):
                  # print(f"Placeholder affiché pour scan {self.current_scan_idx}") # Debug
                  pass # Keep it as the NaN placeholder

        # Update image data
        self.image.set_data(matrix if matrix.size > 0 else np.array([[np.nan]]))
        if self.vmin is not None: # Only set clim if calculation was successful
            self.image.set_clim(vmin=self.vmin, vmax=self.vmax)

        self.update_title()

        # Update scan slider position
        if self.num_scans_current_moment > 1:
            self.slider_scan.eventson = False
            self.slider_scan.set_val(self.current_scan_idx)
            self.slider_scan.eventson = True

        self.fig.canvas.draw_idle()

    def next_scan(self, event=None):
        if self.num_scans_current_moment <= 1: return
        self.current_scan_idx = (self.current_scan_idx + 1) % self.num_scans_current_moment
        self.update_display()

    def prev_scan(self, event=None):
        if self.num_scans_current_moment <= 1: return
        self.current_scan_idx = (self.current_scan_idx - 1 + self.num_scans_current_moment) % self.num_scans_current_moment
        self.update_display()

    def play_pause(self, event=None):
        # (Code logic from previous response - handles play/pause toggle)
        if self.num_scans_current_moment <= 1: return
        if self.is_playing:
            self.is_playing = False; self.btn_play.label.set_text('▶ Play')
            if self.animation_timer is not None: self.animation_timer.stop()
        else:
            self.is_playing = True; self.btn_play.label.set_text('❚❚ Pause')
            if self.animation_timer is None:
                 self.animation_timer = self.fig.canvas.new_timer(interval=DELAI_ANIMATION_MS)
                 self.animation_timer.add_callback(self.animate_step)
            # Ensure timer restarts if already created but stopped
            self.animation_timer.start()
        self.fig.canvas.draw_idle()

    def animate_step(self):
        # (Code logic from previous response - advances frame during play)
        if self.is_playing and self.num_scans_current_moment > 1:
            self.next_scan()
        else:
             if self.animation_timer is not None: self.animation_timer.stop()
             if self.is_playing:
                 self.is_playing = False; self.btn_play.label.set_text('▶ Play'); self.fig.canvas.draw_idle()


# --- Point d'Entrée Principal ---
if __name__ == "__main__":
    # Charger le dictionnaire de matrices groupées par Moment Index
    # Structure: {moment_idx: [matrix_scan_0, matrix_scan_1, ...]}
    donnees_groupees = charger_matrices_par_moment(FICHIER_DATA)

    # Lancer le visualiseur si des données ont été chargées
    if donnees_groupees:
         visualiseur = VisualiseurMoments(donnees_groupees)
    else:
         print("Aucune donnée Moment/Scan n'a été chargée, le visualiseur ne sera pas lancé.")


    print("Fin du script.")
