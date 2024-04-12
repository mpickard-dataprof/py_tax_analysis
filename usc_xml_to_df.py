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

    def _extractSectionInfo(self, node):
        """
        A utility function to extract values of interest from a USC section node.

        Args:
            node (Element): An etree element that is a USC section to extract
            information from.

        Returns:
            dict: A dictionary containing the node id (USC path), the level 
            (which will always be "section"), the section number, the section's 
            parent, and the full section text.
        """
        # make sure we received a section node
        level = self._getLevel(node)
        assert level == "section"

        ## extractElementInfo - returns a dict with the id, level,
        ## level name, and joined texts from nodes of subtree

        # a section node should have an id, level, level_name (i.e., the
        # section number), parent, and text.
        
        # get section number
        label = node[0].text.split()
        if len(label) < 2:
            level_name = label[0]
        else:
            level_name = label[1][:-1]

        return {
            'id': node.attrib['identifier'],
            'level': level,
            'level_name': level_name,
            'parent': self._getParentIdentifier(node),
            'text': self._joinNodeText(node),
        }

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


    def _removeDuplicateText(self, df):
        """A utility function that removes text that duplicate the text of 
        parent levels. The USC Title 26 XML file has <content>, <continuation>,
        and <chapeau> tags that are not logically children of the tags they
        reside in -- by that, I mean they are not true "levels" of the USC
        document. This function removes those duplicate levels and returns
        the deduplicated dataframe.

        Args:
            df (pandas.DataFrame): The Title 26 dataframe generated from parsing
            the XML file. It needs to have 'id', 'parent', 'level' and 'text'
            columns.

        Returns:
            pandas.DataFrame: A dataframe with the duplicated levels dropped.
        """

        # this puts the indices in a column named 'index'
        df.reset_index(inplace=True)

        # parent-to-child is 1-to-many. So 'id' acts as the parent identifier
        # and 'parent' (which is the parent id of the child) acts as the child
        # identifier.  This allows a parent to be matched with many children.
        # (Subtitle A has 7 Chapter children)
        merged_df = df.merge(df, how='right', left_on='id', right_on='parent')

        # create a df containing rows that have the same parent and child text, 
        # where the child is a 'continuation', 'chapeau', or 'content'.
        # The logic here is if a continuation, chapeau, or content node (which
        # are "helper" nodes, not actually logical children) AND it duplicates
        # it's parent's text; then delete it.

        # IMPORTANT: there are instances in real level parent-child relationships
        # where there is the same text. One example include '/us/usc/t26/s101/f/2/A'.
        # It has the same text as it's parent '/us/usc/t26/s101/f/2'.  We want to
        # keep these...that's why we ensure the child is a continuation, chapeau, 
        # or a content.
        sametext_df = merged_df[
            (merged_df['text_x'] == merged_df['text_y']) &
            (merged_df['level_y'].isin(['continuation', 'chapeau', 'content']))
            ]

        # extract the children index ids that are duplicates
        drop_index = sametext_df['index_y'].values

        # drop them from the original df
        return df.drop(drop_index).drop(['index'], axis=1)

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

        df = self._removeDuplicateText(df)
        return df.to_csv("./output/usc26.csv", index_label="row")

    
    # NOTE: It was orders of magnitude faster to generate the section
    # dataframe from the XML rather than reassemble the parts of the
    # section from the dataframe produced by xmlToDataframe().
    
    def xmlToSectionDataframe(self):
        """
        Converts the USC XML format to a Pandas dataframe. Each row of the
        dataframe is a USC section the full text of that section. Saves the 
        CSV file to an 'output' directory, which is created if it doesn't exist.
        """
        # the dictionary list will hold the columns for the dataframe
        dict_list = []

        self._removeTagTypes()

        # just extract section tags
        section = "{"+self._namespace[""]+"}section"
        for element in self._root.iter(tag=section):

            # if the section has been repealed, skip it
            if ("status" in element.attrib):
                if (element.attrib['status'] == 'repealed'):
                    continue

            dict_data = self._extractSectionInfo(element)

            if dict_data != None:
                dict_list.append(dict_data)

        # build and save dataframe
        df = pd.DataFrame.from_dict(dict_list)

        if not os.path.exists("output"):
            os.mkdir("./output")

        return df.to_csv("./output/usc26_sections.csv", index_label="row")
    
# usc = UscReader("./downloads/usc26.xml")
# usc.xmlToSectionDataframe()
