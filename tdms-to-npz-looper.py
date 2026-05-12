import numpy as np
import matplotlib.pyplot as plt
from iqtools import plotters, tools
import os
import logging
from datetime import datetime
import argparse
import toml


# Set logger at the module level so it is accessible everywhere
logger = logging.getLogger(__name__)


# Get list of files in the directory and then sort them (os.listdir returns them in a random order)
def get_filelist(path):
    logger.info(f"Getting file list from {path}")
    datafiles = os.listdir(path)
    datafiles = sorted(datafiles)
    logger.info("Complete")
    # Divide length of files by two to account for the index files
    logger.info(f"{int(len(datafiles) / 2)} files in the directory")

    return datafiles


def get_preexisting_files(path):
    logger.info(f"Getting list of existing output files from {path}")
    datafiles = os.listdir(path)
    datafiles = sorted(datafiles)

    filenumbers = [int(x.split(".")[0]) for x in datafiles]
    logger.info("Complete")

    return filenumbers


def get_settings(filepath, filename, lframes):
    fullpath = filepath + filename
    iq = tools.get_iq_object(fullpath)
    logger.debug(vars(iq))
    freq_bin_size = iq.fs/lframes
    t_bin_size = 1/freq_bin_size
    nframes = int((iq.nsamples_total/iq.fs)/t_bin_size)

    return iq, nframes, freq_bin_size, t_bin_size


def read_data(iq, filename, lframes, nframes):
    logger.info(f"Reading data from {filename}")
    start_time = datetime.now()
    iq.read_complete_file()
    logger.info("Complete")
    end_time = datetime.now()
    time_diff = end_time - start_time
    logger.debug(f"Reading data took {time_diff}")
    logger.debug(f"Total samples: {len(iq.data_array)}")
    logger.debug(f"lframes * nframes: {lframes*nframes}")

    # iq.nsamples_total is hardcoded via the settings of the recording, so if a file was stopped before
    # the recording finished, nsamples_total will not actually match the true number of recorded samples
    if len(iq.data_array) != lframes * nframes:
        raise Exception("nsamples_total does not match the true number of samples in the file, and I don't want to code a solution to this right now. Aborting...")

    return iq


def calculate_spectrogram(iq, nframes, lframes, average_every):
    iq.window = "hamming"
    iq.method = "fftw"
    logger.info("Calculating spectrogram")
    start_time = datetime.now()
    xx, yy, zz = iq.get_power_spectrogram(
        nframes = nframes,
        lframes = lframes,
        sparse = False
    )
    logger.info("Complete")
    end_time = datetime.now()
    time_diff = end_time - start_time
    logger.debug(f"Calculating spectrogram took {time_diff}")

    xx += iq.center

    xx_avg, yy_avg, zz_avg = tools.get_averaged_spectrogram(
        xx = xx,
        yy = yy,
        zz = zz,
        every = average_every
    )

    return xx, zz_avg


def save_spectrum(xx, zz_avg, output_location, output_name):
    zz_avg_sum = np.zeros(np.shape(xx[0]))
    for i in range(np.shape(zz_avg)[0]):
        zz_avg_sum += zz_avg[i]

    logger.info(f"Saving to {output_location}")
    logger.info(f"With file name {output_name}")

    np.savez(
        output_location + output_name,
        xx[0],
        zz_avg_sum
    )


def main():
    parser = argparse.ArgumentParser(
        description="Reads in data from a .tdms file, gets the spectrogram, and saves a 1d spectrum to a .npz file"
    )
    parser.add_argument(
        "--cfg", type=str, required=True, help="Path to the .toml configuration file"
    )

    args = parser.parse_args()
    config_path = args.cfg
    if not os.path.exists(config_path):
        raise FileNotFoundError(f".toml file not found: {config_path}")

    with open(config_path, "r") as f:
        config = toml.load(f)

    file_path = config["settings"]["file_path"]
    starting_file = config["settings"]["starting_file"]
    starting_file = int(starting_file)
    lframes = config["settings"]["lframes"]
    # Convert string e.g. "2**18" to the integer value
    lframes = eval(lframes)
    average_every = config["settings"]["average_every"]
    average_every = int(average_every)
    output_location = config["settings"]["output_location"]
    logging_level = config["settings"]["logging_level"]

    numeric_level = getattr(logging, logging_level.upper(), None)
    logging.basicConfig(level=numeric_level)

    datafiles = get_filelist(file_path)

    logger.info(f"Starting at file number {starting_file}, {int(len(datafiles)/2 - starting_file)} files to process")

    existing_files = get_preexisting_files(output_location)

    for i in range(int((len(datafiles) - starting_file * 2) / 2)):
        filenumber = starting_file * 2 + (i * 2)
        logger.info(f"Filenumber: {filenumber / 2}")
        if filenumber / 2 in existing_files:
            logger.info(f"File with this number has already been processed, skipping to next file...")
            continue

        filename = datafiles[filenumber]
        iq, nframes, freq_bin_size, t_bin_size = get_settings(file_path, filename, lframes)
        logger.info(f"filename: {filename}, nframes: {nframes}, f_res: {freq_bin_size}, t_res: {t_bin_size}")

        output_name = f"{str(filename)}_{str(round(freq_bin_size, 2))}Hz_{str(round(t_bin_size * 1000, 2))}ms_{average_every}avg"

        iq = read_data(iq, filename, lframes, nframes)

        xx, zz_avg = calculate_spectrogram(iq, nframes, lframes, average_every)

        save_spectrum(xx, zz_avg, output_location, output_name)


if __name__ == "__main__":
    main()
