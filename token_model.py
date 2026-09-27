def tokenize_fn(tokenizer, examples, max_length=380):
    # algunas preguntas de SQuAD llevan espacios al principio, lo que desalinea
    # el offset_mapping frente al texto original si no se limpian antes
    questions = [q.lstrip() for q in examples["question"]]

    tokenized = tokenizer(
        questions, examples["context"],
        truncation="only_second", max_length=max_length, padding="max_length",
        return_offsets_mapping=True,
    )

    offset_mapping = tokenized.pop("offset_mapping")
    start_positions, end_positions = [], []

    for i, offsets in enumerate(offset_mapping):
        answer = examples["answers"][i]
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])

        # sequence_ids: None para tokens especiales, 0 para la pregunta, 1 para el contexto
        sequence_ids = tokenized.sequence_ids(i)

        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # si la respuesta cayó fuera de la ventana truncada, se marca con [CLS] (índice 0)
        if offsets[context_start][0] > start_char or offsets[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            idx = context_start
            while idx <= context_end and offsets[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offsets[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    return tokenized


def tokenize_train(tokenizer, examples, max_length=380):
    questions = [q.lstrip() for q in examples["question"]]
    tokenized = tokenizer(
        questions, examples["context"],
        truncation="only_second", max_length=max_length, padding="max_length",
        return_offsets_mapping=True,
    )
    offset_mapping = tokenized.pop("offset_mapping")
    start_positions, end_positions = [], []

    for i, offsets in enumerate(offset_mapping):
        answer = examples["answers"][i]
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])
        sequence_ids = tokenized.sequence_ids(i)

        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        if offsets[context_start][0] > start_char or offsets[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            idx = context_start
            while idx <= context_end and offsets[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offsets[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    return tokenized


def tokenize_val(tokenizer, examples, max_length=380):
    questions = [q.lstrip() for q in examples["question"]]
    tokenized = tokenizer(
        questions, examples["context"],
        truncation="only_second", max_length=max_length, padding="max_length",
        return_offsets_mapping=True,
    )
    # los offsets de los tokens de la pregunta no nos sirven para nada y podrían
    # confundirse con los del contexto al buscar la respuesta -> se anulan
    for i in range(len(questions)):
        sequence_ids = tokenized.sequence_ids(i)
        tokenized["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None
            for k, o in enumerate(tokenized["offset_mapping"][i])
        ]
    return tokenized
