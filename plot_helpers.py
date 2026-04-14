import os


def show_or_autoclose_plot(plt_module, default_pause_seconds=0.5):
    auto_close = os.environ.get("ROSETTA_AUTO_CLOSE_PLOTS", "1").strip().lower()
    if auto_close in {"0", "false", "no", "off"}:
        plt_module.show()
        return

    pause_value = os.environ.get("ROSETTA_AUTO_CLOSE_SECONDS")
    try:
        pause_seconds = max(float(pause_value), 0.0) if pause_value is not None else default_pause_seconds
    except ValueError:
        pause_seconds = default_pause_seconds

    plt_module.show(block=False)
    plt_module.pause(max(pause_seconds, 0.001))
    plt_module.close()
