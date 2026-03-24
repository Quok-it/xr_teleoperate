import triad_openvr
import time
import sys
import curses

v = triad_openvr.triad_openvr()
v.print_discovered_objects()

interval = 1/float(sys.argv[1]) if len(sys.argv) == 2 else 1/250

def run(stdscr):
    curses.curs_set(0)  # hide cursor so it looks less schizo
    while True:
        start = time.time()
        row = 0
        for tracker_name in [k for k in v.devices.keys() if "tracker" in k]:
            pose = v.devices[tracker_name].get_pose_euler()
            if pose is None:
                txt = f"{tracker_name}: not tracked        "
            else:
                txt = f"{tracker_name}: " + " ".join("%.4f" % e for e in pose)
            stdscr.addstr(row, 0, txt)
            row += 1
        stdscr.refresh()
        sleep_time = interval - (time.time() - start)
        if sleep_time > 0:
            time.sleep(sleep_time)

curses.wrapper(run)

