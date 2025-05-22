if __name__ == "__main__":
    try:
        import multiprocessing

        import alphastats  # noqa: F401 TODO check if this can be removed

        multiprocessing.freeze_support()

    except ModuleNotFoundError:
        pass
    from alphastats.gui import run

    run()
