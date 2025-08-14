def demo_progress_usage():
    """
    Placeholder for manual testing of progress UX inside the venv.
    """
    try:
        from panoptes.progress.progress_ux import percent_spinner  # type: ignore
    except Exception:
        print("progress_ux not yet importable; install project in editable mode first.")
        return
    items = ["a", "b", "c", "d"]
    done = 0
    from time import sleep
    with percent_spinner(prefix="DEMO") as sp:
        sp.update(total=len(items))
        for it in items:
            sp.update(current=it)
            sleep(0.2)
            done += 1
            sp.update(count=done)
