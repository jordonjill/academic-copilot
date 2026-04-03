import { useState } from "react";

type Props = {
  disabled?: boolean;
  onSend: (text: string) => Promise<void> | void;
};

export function ChatInput({ disabled, onSend }: Props) {
  const [value, setValue] = useState("");

  async function submit() {
    const trimmed = value.trim();
    if (!trimmed || disabled) {
      return;
    }
    setValue("");
    await onSend(trimmed);
  }

  return (
    <div className="chat-input-wrap">
      <textarea
        value={value}
        onChange={(e) => setValue(e.target.value)}
        rows={4}
        placeholder="输入问题，例如：请给出一个低资源中文法律问答 RAG 的实验设计。"
        onKeyDown={async (e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            await submit();
          }
        }}
      />
      <button type="button" onClick={submit} disabled={disabled || !value.trim()}>
        Send
      </button>
    </div>
  );
}
