import logging


def setup_logger(
    name: str,
    log_file: str = "applog.log",
    level: int = logging.INFO
) -> logging.Logger:
    """
    共通ロガーを作成する関数

    Parameters
    ----------
    name : str
        ロガー名。通常は __name__ を渡す。
    log_file : str
        出力するログファイル名。
    level : int
        ログレベル。logging.INFO, logging.DEBUG など。

    Returns
    -------
    logging.Logger
        設定済みのロガー
    """

    # ロガー作成
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # すでにハンドラが設定されている場合は追加しない
    # 同じログが重複出力されるのを防ぐ
    if logger.handlers:
        return logger

    # ハンドラ設定
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    stream_handler = logging.StreamHandler()

    # ハンドラごとのログレベル設定
    file_handler.setLevel(level)
    stream_handler.setLevel(level)

    # フォーマット設定
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    )

    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # ハンドラをロガーに追加
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger