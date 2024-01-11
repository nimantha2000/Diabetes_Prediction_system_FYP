class Chatbot {
    constructor() {
        this.args = {
            openButton: document.querySelector('.chatbox__button'),
            chatBox: document.querySelector('.chatbox__support'),
            sendButton: document.querySelector('.send__button')
        };

        this.state = false;
        this.messages = [];
    }

    display() {
        const { openButton, chatBox, sendButton } = this.args;

        openButton.addEventListener('click', () => this.toggleState(chatBox));

        sendButton.addEventListener('click', () => this.onSendButton(chatBox));

        const node = chatBox.querySelector('input');
        node.addEventListener("keyup", (event) => {
            if (event.key === "Enter") {
                this.onSendButton(chatBox);
            }
        });
    }

    toggleState(chatbox) {
        this.state = !this.state;

        // show or hide the box
        if (this.state) {
            chatbox.classList.add('chatbox--active');
        } else {
            chatbox.classList.remove('chatbox--active');
        }
    }

    onSendButton(chatbox) {
        const textField = chatbox.querySelector('input');
        let text1 = textField.value;
        if (text1 === "") {
            return;
        }

        let customerMessage = { name: "customer", message: text1 };
        this.messages.push(customerMessage);

        console.log("Sending message:", text1);

        fetch($SCRIPT_ROOT + '/predict', {
            method: 'POST',
            body: JSON.stringify({ message: text1 }),
            mode: 'cors',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(r => {
            console.log("Bot Response:", r);
            let botMessage = { name: "bot", message: r.answer };
            this.messages.push(botMessage);
            this.updateChatText(chatbox);
            textField.value = '';
        }).catch((error) => {
            console.error('Error:', error);
            this.updateChatText(chatbox);
            textField.value = '';
        });
    }

    updateChatText(chatbox) {
        let html = '';
        // Reverse the order of messages
        const reversedMessages = this.messages.slice().reverse();

        reversedMessages.forEach(function(item) {
            if (item.name === "customer") {
                html += '<div class="message__item message__item--customer right">' + item.message + '</div>';
            } else {
                html += '<div class="message__item message__item--bot left">' + item.message + '</div>';
            }
        });

        const chatmessage = chatbox.querySelector('.chatbox__messages');
        chatmessage.innerHTML = html;
    }
}

const chatbox = new Chatbot();
chatbox.display();
