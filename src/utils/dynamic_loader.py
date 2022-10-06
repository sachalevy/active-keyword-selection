import typing

import orjson


class MongoExportIterator:
    """
    Iterate over a json document from a file descriptor. It is assumed that the
    file is structured a list of dicts, each dict being the lowest level structure to parse.
    Individual dicts are parsed based on curly braces indentation. Lines begining with `},{`
    are identified as begining of a new document.

    Note that this corresponds to the export format for MongoDB collections.

    :param fd: Open file to read raw json from.
    :type fd: typing.IO
    """

    def __init__(self, fd: typing.IO = None):
        self.fd = fd
        self.line_count = 0
        self.item_count = 0
        self.tmp = ""
        self.start_of_doc_marker = "{"
        self.end_of_doc_marker = "},{"
        self.end_of_list_marker = "}]"
        self.done = False

    def clean_line(self, line: str) -> str:
        """Preprocess line string for end of tweet checker.

        :param line: string line
        :type line: str
        :return: processed line string
        :rtype: str
        """
        return line.replace("\n", "")

    def __next__(
        self,
    ):
        completed_doc_read = False
        while not completed_doc_read and not self.done:
            line = next(self.fd).strip(" \n")
            if line == self.end_of_doc_marker:
                self.tmp += "}"
                completed_doc_read = True
            elif line == self.end_of_list_marker:
                self.tmp += "}"
                completed_doc_read = True
                self.done = True
            else:
                self.tmp += line
                self.item_count += 1
            self.line_count += 1

        output = self.tmp
        self.tmp = self.start_of_doc_marker

        if not self.done:
            return orjson.loads(output)
        else:
            raise StopIteration()

    def __iter__(self):
        if self.line_count == 0:
            # initialize read of the first document
            line = next(self.fd)
            assert self.start_of_doc_marker == line.replace("[", "").strip()
            self.tmp += self.start_of_doc_marker
            self.line_count += 1

        return self
