import { Example } from "./Example";

import styles from "./Example.module.css";

export type ExampleModel = {
    text: string;
    value: string;
};

const EXAMPLES: ExampleModel[] = [
    {
        text: "ttb all free คืออะไร",
        value: "ttb all free คืออะไร"
    },
    { text: "ttb all free ปัจจุบันมีสิทธิ์ได้ประกันอุบัติเหตุฟรีหรือไม่", value: "ลูกค้า all free ปัจจุบันมีสิทธิ์ได้ประกันอุบัติเหตุฟรีหรือไม่" },
    { text: "ความคุ้มครอง ของ ttb all free ที่ได้รับสามารถคุ้มครองอุบัติเหตุอะไรได้บ้าง", value: "ความคุ้มครอง ของ all free ที่ได้รับสามารถคุ้มครองอุบัติเหตุอะไรได้บ้าง" },
];

interface Props {
    onExampleClicked: (value: string) => void;
}

export const ExampleList = ({ onExampleClicked }: Props) => {
    return (
        <ul className={styles.examplesNavList}>
            {EXAMPLES.map((x, i) => (
                <li key={i}>
                    <Example text={x.text} value={x.value} onClick={onExampleClicked} />
                </li>
            ))}
        </ul>
    );
};
