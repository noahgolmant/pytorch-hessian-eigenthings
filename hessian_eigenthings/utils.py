""" small helpers """
import logging
import shutil
import sys
import time

TOTAL_BAR_LENGTH = 65.0

term_width = shutil.get_terminal_size().columns


def log(msg):
    logging.info("[hessian_eigenthings] " + str(msg))


def maybe_fp16(vec, fp16):
    return vec.half() if fp16 else vec.float()


last_time = time.time()
begin_time = last_time


def format_time(seconds):
    """ converts seconds into day-hour-minute-second-ms string format """
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    formatted = ""
    i = 1
    if days > 0:
        formatted += str(days) + "D"
        i += 1
    if hours > 0 and i <= 2:
        formatted += str(hours) + "h"
        i += 1
    if minutes > 0 and i <= 2:
        formatted += str(minutes) + "m"
        i += 1
    if secondsf > 0 and i <= 2:
        formatted += str(secondsf) + "s"
        i += 1
    if millis > 0 and i <= 2:
        formatted += str(millis) + "ms"
        i += 1
    if formatted == "":
        formatted = "0ms"
    return formatted


def progress_bar(current, total, msg=None):
    """handy utility to display an updating progress bar...
    percentage completed is computed as current/total

    from: https://github.com/noahgolmant/skeletor/blob/master/skeletor/utils.py
    """
    global last_time, begin_time  # pylint: disable=global-statement
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(" [")
    for _ in range(cur_len):
        sys.stdout.write("=")
    sys.stdout.write(">")
    for _ in range(rest_len):
        sys.stdout.write(".")
    sys.stdout.write("]")

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    msg_parts = []
    msg_parts.append("  Step: %s" % format_time(step_time))
    msg_parts.append(" | Tot: %s" % format_time(tot_time))
    if msg:
        msg_parts.append(" | " + msg)

    msg = "".join(msg_parts)
    sys.stdout.write(msg)
    for _ in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(" ")

    # Go back to the center of the bar.
    for _ in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write("\b")
    sys.stdout.write(" %d/%d " % (current + 1, total))

    if current < total - 1:
        sys.stdout.write("\r")
    else:
        sys.stdout.write("\n")
    sys.stdout.flush()
