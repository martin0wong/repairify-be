import { twMerge } from "tailwind-merge"

interface props {
    className: string | undefined
}

export default function PromptContainer({ className } : props) {
    return (
        <>
        {/* Prompt Messages Container - Modify the height according to your need */}
        <div className={twMerge('flex h-[80vh] w-full flex-col',className)}>
        {/* Prompt Messages */}
        <div
            className="flex-1 space-y-6 overflow-y-auto rounded-xl bg-slate-200 p-4 text-sm leading-6 text-slate-900 shadow-sm dark:bg-slate-900 dark:text-slate-300 sm:text-base sm:leading-7"
        >
            <div className="flex items-start">
            <img
                className="mr-2 h-8 w-8 rounded-full"
                src="https://dummyimage.com/128x128/363536/ffffff&text=J"
            />
            <div
                className="flex rounded-b-xl rounded-tr-xl bg-slate-50 p-4 dark:bg-slate-800 sm:max-w-md md:max-w-2xl"
            >
                <p>Explain quantum computing in simple terms</p>
            </div>
            </div>
            <div className="flex flex-row-reverse items-start">
            <img
                className="ml-2 h-8 w-8 rounded-full"
                src="https://dummyimage.com/128x128/354ea1/ffffff&text=G"
            />

            <div
                className="flex min-h-[85px] rounded-b-xl rounded-tl-xl bg-slate-50 p-4 dark:bg-slate-800 sm:min-h-0 sm:max-w-md md:max-w-2xl"
            >
                <p>
                Certainly! Quantum computing is a new type of computing that relies on
                the principles of quantum physics. Traditional computers, like the one
                you might be using right now, use bits to store and process
                information. These bits can represent either a 0 or a 1. In contrast,
                quantum computers use quantum bits, or qubits.<br /><br />
                Unlike bits, qubits can represent not only a 0 or a 1 but also a
                superposition of both states simultaneously. This means that a qubit
                can be in multiple states at once, which allows quantum computers to
                perform certain calculations much faster and more efficiently
                </p>
            </div>
            </div>
            <div className="flex items-start">
            <img
                className="mr-2 h-8 w-8 rounded-full"
                src="https://dummyimage.com/128x128/363536/ffffff&text=J"
            />
            <div
                className="flex rounded-b-xl rounded-tr-xl bg-slate-50 p-4 dark:bg-slate-800 sm:max-w-md md:max-w-2xl"
            >
                <p>What are three great applications of quantum computing?</p>
            </div>
            </div>
            <div className="flex flex-row-reverse items-start">
            <img
                className="ml-2 h-8 w-8 rounded-full"
                src="https://dummyimage.com/128x128/354ea1/ffffff&text=G"
            />
            <div
                className="flex min-h-[85px] rounded-b-xl rounded-tl-xl bg-slate-50 p-4 dark:bg-slate-800 sm:min-h-0 sm:max-w-md md:max-w-2xl"
            >
                <p>
                Three great applications of quantum computing are: Optimization of
                complex problems, Drug Discovery and Cryptography.
                </p>
            </div>

            </div>
            </div>
            {/* Prompt message input */}
            <form className="mt-2">
                <label htmlFor="chat-input" className="sr-only">Enter your prompt</label>
                <div className="relative">
                <button
                    type="button"
                    className="absolute inset-y-0 left-0 flex items-center pl-3 text-slate-500 hover:text-blue-600 dark:text-slate-400 dark:hover:text-blue-600"
                >
                    <svg
                    aria-hidden="true"
                    className="h-5 w-5"
                    viewBox="0 0 24 24"
                    xmlns="http://www.w3.org/2000/svg"
                    strokeWidth="2"
                    stroke="currentColor"
                    fill="none"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    >
                    <path stroke="none" d="M0 0h24v24H0z" fill="none"></path>
                    <path
                        d="M9 2m0 3a3 3 0 0 1 3 -3h0a3 3 0 0 1 3 3v5a3 3 0 0 1 -3 3h0a3 3 0 0 1 -3 -3z"
                    ></path>
                    <path d="M5 10a7 7 0 0 0 14 0"></path>
                    <path d="M8 21l8 0"></path>
                    <path d="M12 17l0 4"></path>
                    </svg>
                    <span className="sr-only">Use voice input</span>
                </button>
                <textarea
                    id="chat-input"
                    className="block w-full resize-none rounded-xl border-none bg-slate-200 p-4 pl-10 pr-20 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-blue-600 dark:bg-slate-900 dark:text-slate-200 dark:placeholder-slate-400 dark:focus:ring-blue-600 sm:text-base"
                    placeholder="Enter your prompt"
                    rows={1}
                    required
                ></textarea>
                <button
                    type="submit"
                    className="absolute bottom-2 right-2.5 rounded-lg bg-blue-700 px-4 py-2 text-sm font-medium text-slate-200 hover:bg-blue-800 focus:outline-none focus:ring-4 focus:ring-blue-300 dark:bg-blue-600 dark:hover:bg-blue-700 dark:focus:ring-blue-800 sm:text-base"
                >
                    Send <span className="sr-only">Send message</span>
                </button>
                </div>
            </form>
        </div>
        </>
    )
}