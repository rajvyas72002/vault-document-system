from app.extraction import order_blocks

W = 600
LONG = "word " * 40  # > 120 characters


def block(x0, y0, x1, text):
    return (x0, y0, x1, y0 + 10, text, 0, 0)


def texts(blocks):
    return [b[4] for b in blocks]


def test_two_columns_are_read_one_after_the_other():
    # sorted top-to-bottom the way PyMuPDF returns them: the columns are interleaved
    interleaved = [
        block(80, 100, 280, "L1 " + LONG), block(330, 105, 530, "R1 " + LONG),
        block(80, 200, 280, "L2 " + LONG), block(330, 205, 530, "R2 " + LONG),
    ]
    got = [t.split()[0] for t in texts(order_blocks(interleaved, W))]
    assert got == ["L1", "L2", "R1", "R2"]


def test_full_width_heading_and_footer_stay_in_place():
    page = [
        block(50, 20, 550, "HEADING"),
        block(80, 100, 280, "L1 " + LONG), block(330, 105, 530, "R1 " + LONG),
        block(80, 200, 280, "L2 " + LONG), block(330, 205, 530, "R2 " + LONG),
        block(250, 700, 350, "FOOTER"),
    ]
    got = [t.split()[0] for t in texts(order_blocks(page, W))]
    assert got == ["HEADING", "L1", "L2", "R1", "R2", "FOOTER"]


def test_single_column_page_is_left_alone():
    page = [block(50, 100, 550, "A " + LONG), block(50, 200, 550, "B " + LONG), block(50, 300, 550, "C " + LONG)]
    assert order_blocks(page, W) == page


def test_table_cells_are_not_mistaken_for_columns():
    cells = [block(50, 100, 250, "Plan"), block(350, 100, 550, "Price"),
             block(50, 130, 250, "Gold"), block(350, 130, 550, "100 per month")]
    assert order_blocks(cells, W) == cells
