if __name__ == "__main__":
    try:
        import multiprocessing

        import alphastats

        multiprocessing.freeze_support()

    except ModuleNotFoundError:
        pass
    from alphastats.gui import run

    run()
