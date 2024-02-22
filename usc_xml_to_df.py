from lxml import etree
import pandas as pd

# define the namespace
namespace = {'': "http://xml.house.gov/schemas/uslm/1.0",
             'xml' : "http://www.w3.org/1999/xhtml"}

# define levels to visit
levels = ['subtitle', 'chapter', 'subchapter', 'part', 'subpart', 'section', 
          'subsection', 'paragraph', 'chapeau', 'continuation', 'content', 'table', 'subparagraph', 'clause', 
          'subclause', 'item', 'subitem', 'subsubitem']

levels = ['{'+namespace['']+'}'+level for level in levels]

# the dictionary list will hold the columns for the dataframe
dict_list = []

def joinNodeText(node):
    ## joinNodeText iterates over all text nodes of an Element
    ## and joins the text into a single string
    return ' '.join([t.strip() for t in node.itertext() if t.strip() != ""])

def getParentIdentifier(node):
    if('identifier' not in node.getparent().attrib.keys()):
        return None
    else:
        return node.getparent().attrib['identifier']

def extractElementInfo(node):
    ## extractElementInfo - returns a dict with the id, level, 
    ## level name, and joined texts from nodes of subtree
    
    ## -- get identifier, which is different from id
    ## the identifier is a path to the node
    if ('identifier' not in node.attrib.keys()):
        # if there is no id, chances are there is no text or
        ## no level name (i.e., .tag), so skipping these nodes
        print("did not find an identifier in: " + node.tag + "...")
        ## if there's not identifier than we are processing a 'chapeau',
        ## 'content', 'table', or 'continuation' element, so just extract
        ## the text
        return {
            'id': None,
            'level': getLevel(node),
            'level_name': None,
            'parent': getParentIdentifier(node),
            'text': joinNodeText(node)
        }
    else:
        id = node.attrib['identifier']

    ## -- get level (e.g., 'section', 'subsection', etc)
    level = getLevel(node)
    
    ## -- get level name 
    label = node[0].text.split()
    if (len(label) < 2):
        level_name = label[0]
    else:
        level_name = label[1][:-1]

    ## -- get heading (if there is one)
    if (node[1].text):
        text = node[1].text.strip()
    else:
        text = None

    dict_info = {
        'id': id,
        'level': level,
        'level_name': level_name,
        'parent': getParentIdentifier(node),
        'text': text
    }

    return dict_info

def getLevel(node):
    assert(node.tag)
    level = node.tag.split('}')[1]
    return level

# these are the parts we want to remove
# I'm considering these "not part of USC 26" (they are meta information)
tags = ['note', 'notes', 'toc', 'meta', 'sourceCredit']

def removeTagTypes(tagList, xmlNode, ns):
    for tag in tagList:
        nodeList = xmlNode.findall(".//" + tag, ns)
        for node in nodeList:
            parent = node.getparent()
            parent.remove(node)

# parse the XML file
tree = etree.parse("./downloads/usc26.xml")
root = tree.getroot()

# Step #1: remove "meta" parts
removeTagTypes(tags, root, namespace)



for element in root.iter(tag = levels):
    dict_data = extractElementInfo(element)
    
    if (dict_data != None):
        dict_list.append(dict_data)

# build and save dataframe
df = pd.DataFrame.from_dict(dict_list)
df.to_csv("./usc26.csv")