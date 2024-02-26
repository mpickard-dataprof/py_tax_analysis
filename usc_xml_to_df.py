from lxml import etree
import pandas as pd
import os.path


class UscReader:

    # protected members
    def __init__(self, filepath) -> None:
        # xml elements won't be found without namespaces
        self._namespace = {
            "": "http://xml.house.gov/schemas/uslm/1.0",
            # "xml": "http://www.w3.org/1999/xhtml",
        }

        self._levels = [
            "subtitle",
            "chapter",
            "subchapter",
            "part",
            "subpart",
            "section",
            "subsection",
            "paragraph",
            # <Start: not USC levels, but contain text
            "chapeau",
            "continuation",
            "content",
            "table",
            # End>
            "subparagraph",
            "clause",
            "subclause",
            "item",
            "subitem",
            "subsubitem",
        ]

        # add default namespace on the front of the levels
        self._levels = [
            "{" + self._namespace[""] + "}" + level for level in self._levels
        ]

        # these are the parts we want to remove
        # I'm considering these "not part of USC 26" (they are meta information)
        self._tags_to_remove = ["note", "notes", "toc", "meta", "sourceCredit"]

        # parse the XML file
        tree = etree.parse(filepath)
        self._root = tree.getroot()

    def _joinNodeText(self, node):
        """
        A utility function to extract the text found in a node. This will 
        extract and join all the text found in this node and its children
        nodes.

        Args:
            node (Element): An etree Element node.

        Returns:
            str: A concatenated and cleaned string.
        """
        ## joinNodeText iterates over all text nodes of an Element
        ## and joins the text into a single string
        return " ".join([t.strip() for t in node.itertext() if t.strip() != ""])

    def _getParentIdentifier(self, node):
        """
        A utility function to return the parent of the etree element node.

        Args:
            node (Element): An etree Element object.

        Returns:
            str: The USC path of the parent of the node.
        """

        if "identifier" not in node.getparent().attrib.keys():
            return None
        else:
            return node.getparent().attrib["identifier"]

    def _extractElementInfo(self, node):
        """
        A utility function to extract values of interest from the USC node.

        Args:
            node (Element): An etree element to extract the USC node information
            from.

        Returns:
            dict: A dictionary containing the node id (USC path), the level,
            the level value, the node's parent, and the text specific to that
            level.
        """
        ## extractElementInfo - returns a dict with the id, level,
        ## level name, and joined texts from nodes of subtree

        ## -- get identifier, which is different from id
        ## the identifier is a path to the node
        if "identifier" not in node.attrib.keys():
            ## if there's not identifier than we are processing a 'chapeau',
            ## 'content', 'table', or 'continuation' element, so just extract
            ## the text
            return {
                "id": None,
                "level": self._getLevel(node),
                "level_name": None,
                "parent": self._getParentIdentifier(node),
                "text": self._joinNodeText(node),
            }
        else:
            ## if there is an identifier, we are processing a USC level node
            ## so we just want to extract the USC path.
            id = node.attrib["identifier"]

        ## -- get level (e.g., 'section', 'subsection', etc)
        level = self._getLevel(node)

        ## -- get level name
        label = node[0].text.split()
        if len(label) < 2:
            level_name = label[0]
        else:
            level_name = label[1][:-1]

        ## -- get heading (if there is one)
        if node[1].text:
            text = node[1].text.strip()
        else:
            text = None

        dict_info = {
            "id": id,
            "level": level,
            "level_name": level_name,
            "parent": self._getParentIdentifier(node),
            "text": text,
        }

        return dict_info

    def _getLevel(self, node):
        """
        A utility function to return the level (chapter, section, subsection,
        etc) from the node.

        Args:
            node (Element): The etree element to extract the level info from.

        Returns:
            str: The level value of the USC26 node.
        """
        assert node.tag
        level = node.tag.split("}")[1]
        return level

    # I'm assuming we'll always want to remove the same tags,
    # so the tag list is a protected member.
    def _removeTagTypes(self):
        """
        Utility function to remove tags (elements) that act as metadata
        to the USC; these include notes, table of contents, and credits.
        """
        for tag in self._tags_to_remove:
            nodeList = self._root.findall(".//" + tag, self._namespace)
            for node in nodeList:
                parent = node.getparent()
                parent.remove(node)

    def xmlToDataframe(self):
        """
        Converts the USC XML format to a Pandas dataframe. Each row of the
        dataframe is a USC level (e.g., chapter, section, subsection, etc.)
        with the text associated with that level. Saves the CSV file to
        an 'output' directory, which is created if it doesn't exist

        """
        # the dictionary list will hold the columns for the dataframe
        dict_list = []

        self._removeTagTypes()

        for element in self._root.iter(tag=self._levels):
            dict_data = self._extractElementInfo(element)

            if dict_data != None:
                dict_list.append(dict_data)

        # build and save dataframe
        df = pd.DataFrame.from_dict(dict_list)

        if not os.path.exists("output"):
            os.mkdir("./output")

        df.to_csv("./output/usc26.csv", index_label="row")


usc = UscReader("./downloads/usc26.xml")
usc.xmlToDataframe()
