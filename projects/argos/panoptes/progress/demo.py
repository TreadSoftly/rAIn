def demo_progress_usage():
    """
    Placeholder for manual testing of progress UX inside the venv.
    Uses the HALO spinner directly.
    """
    try:
        from panoptes.progress import percent_spinner  # type: ignore
    except Exception:
        print("HALO spinner not yet importable; install the project in editable mode first.")
        return
    items = ["a", "b", "c", "d"]
    done = 0
    from time import sleep
    with percent_spinner(prefix="DEMO") as sp:
        sp.update(total=len(items))
        for it in items:
            sp.update(item=it)  # [File: …]
            sleep(0.2)
            done += 1
            sp.update(count=done, job="demo")  # [Job: demo]
