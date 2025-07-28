from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage
    )


# Create a new vector index for the files provided.
def InitializeIndex(persistPath, sourcePath, docName):
    # load data
    example_file_1 = SimpleDirectoryReader(
        input_files=[sourcePath]
    ).load_data()

    # build index
    index = VectorStoreIndex.from_documents(docName, show_progress=True)

    # persist index
    index.storage_context.persist(persist_dir=persistPath)

    return index


def LoadIndex(persistPath, sourcePath, docName):
    try:
        storage_context = StorageContext.from_defaults(
            persist_dir=persistPath
        )
        index = load_index_from_storage(storage_context)

    except:
        index = InitializeIndex(sourcePath, persistPath, docName)
    return index