from lxml import etree
import pandas as pd
import stanza

# define the namespace
ns = {'': "http://xml.house.gov/schemas/uslm/1.0",}

# the dictionary list will hold the columns for the dataframe
dict_list = []

def joinNodeText(node):
    ## joinNodeText iterates over all text nodes of an Element
    ## and joins the text into a single string
    return ' '.join([t.strip() for t in node.itertext() if t.strip() != ""])


def extractElementInfo(node):
    ## -- get id
    ## extractElementInfo - returns a dict with the id, level, 
    ## level name, and joined texts from nodes of subtree
    if ('identifier' not in node.attrib.keys()):
        # if there is no id, chances are there is no text or
        ## no level name (i.e., .tag), so skipping these nodes
        return None
    else:
        id = node.attrib['identifier']

    ## -- get level
    assert(node.tag)
    level = node.tag.split('}')[1]
    
    ## -- get level name
    label = node[0].text.split()
    if (len(label) < 2):
        level_name = label[0]
    else:
        level_name = label[1][:-1]

    ## -- get text
    # extract complete text for subsections only
    # for all other higher level, extract only the 
    # heading text
    if (level == 'subsection'):
        text = joinNodeText(node)
    else: 
        text = node[1].text

    dict_info = {
        'id': id,
        'level': level,
        'level_name': level_name,
        'text': text
    }

    return dict_info




# these are the parts we want to remove
# I'm considering these "not part of USC 26" (they are meta information)
tags = ['note', 'notes', 'toc', 'meta', 'sourceCredit']

def removeTagTypes(tagList, xmlNode, namespace):
    for tag in tagList:
        nodeList = xmlNode.findall(".//" + tag, namespace)
        for node in nodeList:
            parent = node.getparent()
            parent.remove(node)

# parse the XML file
tree = etree.parse("/data/rstudio/corpi/downloads/usc26.xml")
root = tree.getroot()

# remove "meta" parts
removeTagTypes(tags, root, ns)

## !!NOTE:!! Only go down to level of 'subsection' things get messy with text nodes
## below 'subsection'
levels = ['subtitle', 'chapter', 'subchapter', 'part', 'subpart', 'section', 
            'subsection']

namespace = '{http://xml.house.gov/schemas/uslm/1.0}'

levels = [namespace+level for level in levels]

# extract nodes down to subsection
for element in root.iter(tag = levels):
    dict_data = extractElementInfo(element)
    
    if (dict_data != None):
        dict_list.append(dict_data)

# build and save dataframe
df = pd.DataFrame.from_dict(dict_list)
df.to_csv("./usc26.csv")