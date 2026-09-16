import os
import numpy as np
import pandas as pd
import wfdb
import sys

ruta_base = ".."
sys.path.append(os.path.abspath(ruta_base))

from models.ctg import CTG
from models.df_ctg import DF_CTG


def read_data(
    path_folder_data: str, cut_time: int, freq: int, private_db: bool = False
) -> DF_CTG:
    """
    Read CTG signal data and clinical metadata from files.

    This function reads `.dat` and `.hea` files containing fetal monitoring signals and
    corresponding clinical data. Only the last `cut_time` minutes of each signal are retained.

    Args:
        path_folder_data (str): Path to folder containing CTG files.
        cut_time (int): Number of minutes to retain from the end of each signal.
        freq (int): Sampling frequency of the signal (e.g., 4Hz = 4 samples per second).

    Returns:
        DF_CTG: Object containing loaded CTG signals and associated clinical metadata.

    Notes:
        - Files without a `.dat` or `.hea` counterpart are skipped.
        - Clinical metadata is manually parsed from `.hea` file comments.
        - Only `.dat` files are processed (with a matching `.hea`).
    """

    # TODO: Hacer que encuentre la frecuencia sin pasar por parámetro

    # We create an instance of the df_ctg class
    df = DF_CTG(freq=freq)

    # Variables to provide information on the progress of data reading
    num_files_no_exist = 0
    num_ctg_created = 0
    num_cut_rows = freq * cut_time * 60 # 4 datos/sec * 60 sec/min * x min

    # Load private database
    if private_db:
        print("Reading the database of the October 12 hospital")

        fhr_12O = pd.read_csv(
            path_folder_data + "fhr.csv", index_col=0, compression="gzip"
        )

        uc_12O = pd.read_csv(
            path_folder_data + "uc.csv", index_col=0, compression="gzip"
        )

        clinical_12O = pd.read_csv(
            path_folder_data + "clinical.csv", index_col=0, compression="gzip"
        )

        for index in fhr_12O.index:

            if (index in uc_12O.index) and (index in clinical_12O.index):
                single_fhr_12O = fhr_12O.loc[index]
                single_uc_12O = uc_12O.loc[index]
                single_ph = float(clinical_12O.loc[index]["PH"])

                df.new_ctg(single_fhr_12O, single_uc_12O, single_ph, index)
                print(f"{index} created!")
                num_ctg_created += 1

            elif (index not in uc_12O.index) and (index not in clinical_12O.index):
                num_files_no_exist = num_files_no_exist + 1

                print(
                    f"{num_files_no_exist}.- The uc and the clinical information for the ctg ({index}) does not exist. Skipping iteration."
                )

            elif index not in uc_12O.index:
                num_files_no_exist = num_files_no_exist + 1

                print(
                    f"{num_files_no_exist}.- The uc for the ctg ({index}) does not exist. Skipping iteration."
                )

            else:
                num_files_no_exist = num_files_no_exist + 1

                print(
                    f"{num_files_no_exist}.- The clinical information for the ctg ({index}) does not exist. Skipping iteration."
                )

    # Load public database
    else:
        num_files = len(  # TODO: Cambiarlo mejor al numero de ficheros .dat (o .hea)
            [
                f
                for f in os.listdir(path_folder_data)
                if os.path.isfile(os.path.join(path_folder_data, f))
            ]
        )
        print("There are", num_files, "files in this directory")

        # Iterates through the documents in the directory
        for file in os.listdir(path_folder_data):

            # Vamos a leer solo los .dat
            if file.lower().endswith(".hea"):
                continue

            name_file = file.split(".")[0]
            file_dat = name_file + ".dat"
            file_hea = name_file + ".hea"

            # Check that both files exist
            if not os.path.exists(path_folder_data + file_dat) or not os.path.exists(
                path_folder_data + file_hea
            ):
                num_files_no_exist += 1

                if not os.path.exists(path_folder_data + file_dat):
                    print(
                        f"{num_files_no_exist}.- The file ({file_dat}) does not exist. Skipping iteration."
                    )

                elif not os.path.exists(path_folder_data + file_hea):
                    print(
                        f"{num_files_no_exist}.- The file ({file_hea}) does not exist. Skipping iteration."
                    )

                continue

            # We increase the counter of files we load
            num_ctg_created += 1

            # We read the .dat file, contains information on the FHR and UC
            print(f"Loading: {file_dat}")
            record = wfdb.rdrecord(os.path.join(path_folder_data, name_file))

            # We pass the CTG information to a data frame and cut it to the number of minutes passed as a parameter.
            fhr_uc = pd.DataFrame(record.p_signal, columns=record.sig_name).tail(
                num_cut_rows
            )

            # We extract fhr and uc, and give them as indices the real time in minutes of the signal
            fhr = fhr_uc["FHR"]
            fhr.index = np.arange(len(fhr)) / (freq)

            uc = fhr_uc["UC"]
            uc.index = np.arange(len(uc)) / (freq)

            # We define the file name as id, this name is unique
            id = float(name_file)

            # We read the .hea file, contains clinical information
            print(f"Loading: {file_hea}")

            # Categories within the .hea file
            index = [
                "ph",
                "bdecf",
                "pco2",
                "be",
                "apgar1",
                "apgar5",
                "gest. weeks",
                "weight",
                "sex",
                "age",
                "gravidity",
                "parity",
                "diabetes",
                "hypertension",
                "preeclampsia",
                "liq. praecox",
                "pyrexia",
                "meconium",
                "presentation",
                "induced",
                "I. stage",
                "noProgress",
                "CK/KP",
                "II. stage",
                "deliv. type",
            ]

            # We open the file and save the information for each of the categories
            with open(path_folder_data + file_hea, "r") as file_hea:
                for line in file_hea:
                    if "#pH" in line:
                        ph = str(line.split()[1])
                    elif "#BDecf" in line:
                        bdecf = str(line.split()[1])
                    elif "#pCO2" in line:
                        pco2 = str(line.split()[1])
                    elif "#BE" in line:
                        be = str(line.split()[1])
                    elif "#Apgar1" in line:
                        apgar1 = str(line.split()[1])
                    elif "#Apgar5" in line:
                        apgar5 = str(line.split()[1])
                    elif "#Gest. weeks" in line:
                        weeks = str(line.split()[2])
                    elif "#Weight(g)" in line:
                        weight = str(line.split()[1])
                    elif "#Sex" in line:
                        sex = str(line.split()[1])
                    elif "#Age" in line:
                        age = str(line.split()[1])
                    elif "#Gravidity" in line:
                        gravidity = str(line.split()[1])
                    elif "#Parity" in line:
                        parity = str(line.split()[1])
                    elif "#Diabetes" in line:
                        diabetes = str(line.split()[1])
                    elif "#Hypertension" in line:
                        hypertension = str(line.split()[1])
                    elif "#Preeclampsia" in line:
                        preeclampsia = str(line.split()[1])
                    elif "#Liq. praecox" in line:
                        liq_praecox = str(line.split()[2])
                    elif "#Pyrexia" in line:
                        pyrexia = str(line.split()[1])
                    elif "#Meconium" in line:
                        meconium = str(line.split()[1])
                    elif "#Presentation" in line:
                        presentation = str(line.split()[1])
                    elif "#Induced" in line:
                        induced = str(line.split()[1])
                    elif "#I.stage" in line:
                        I_stage = str(line.split()[1])
                    elif "#NoProgress" in line:
                        no_progress = str(line.split()[1])
                    elif "#CK/KP" in line:
                        ck_kp = str(line.split()[1])
                    elif "#II.stage" in line:
                        II_stage = str(line.split()[1])
                    elif "#Deliv. type" in line:
                        deliv_type = str(line.split()[2])

            # We convert the information to a data frame
            data = np.array(
                [
                    ph,
                    bdecf,
                    pco2,
                    be,
                    apgar1,
                    apgar5,
                    weeks,
                    weight,
                    sex,
                    age,
                    gravidity,
                    parity,
                    diabetes,
                    hypertension,
                    preeclampsia,
                    liq_praecox,
                    pyrexia,
                    meconium,
                    presentation,
                    induced,
                    I_stage,
                    no_progress,
                    ck_kp,
                    II_stage,
                    deliv_type,
                ]
            )
            clinical_data = pd.DataFrame(data=data, index=index)

            # We pass the information to df_ctg, within that class a ctg object will be created and stored
            ph = float(clinical_data.loc["ph"][0])
            df.new_ctg(fhr, uc, ph, id)

    # We show the number of files loaded to check if the loading has been performed correctly.
    print(num_ctg_created, "ctg created!")
    return df
