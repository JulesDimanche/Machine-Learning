{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyOeqiH0xZNiAw39+7tiNN7A",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/JulesDimanche/Machine-Learning/blob/main/Spam_detection.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "id": "L3pl0Y3BWRJi"
      },
      "outputs": [],
      "source": [
        "import pandas as pd"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "messages=pd.read_csv('/content/SMSSpamCollection.txt',sep='\\t',names=['label','message'])#tab separated"
      ],
      "metadata": {
        "id": "FrxcOm9EWcz7"
      },
      "execution_count": 2,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "len(messages)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "e97NWvJka73N",
        "outputId": "c8e51816-2012-4ff7-f583-5a88020c31e6"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "5572"
            ]
          },
          "metadata": {},
          "execution_count": 5
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "messages['message'][1]"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 35
        },
        "id": "EnSoKS2fbFVj",
        "outputId": "485b569c-ea75-4c5d-b9ab-e7ceac02ec02"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "'Ok lar... Joking wif u oni...'"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "string"
            }
          },
          "metadata": {},
          "execution_count": 6
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "messages"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 423
        },
        "id": "-eudz9SKXrxU",
        "outputId": "f3ab0eb7-a2e0-4119-a869-c48fb2b71c77"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "     label                                            message\n",
              "0      ham  Go until jurong point, crazy.. Available only ...\n",
              "1      ham                      Ok lar... Joking wif u oni...\n",
              "2     spam  Free entry in 2 a wkly comp to win FA Cup fina...\n",
              "3      ham  U dun say so early hor... U c already then say...\n",
              "4      ham  Nah I don't think he goes to usf, he lives aro...\n",
              "...    ...                                                ...\n",
              "5567  spam  This is the 2nd time we have tried 2 contact u...\n",
              "5568   ham               Will ü b going to esplanade fr home?\n",
              "5569   ham  Pity, * was in mood for that. So...any other s...\n",
              "5570   ham  The guy did some bitching but I acted like i'd...\n",
              "5571   ham                         Rofl. Its true to its name\n",
              "\n",
              "[5572 rows x 2 columns]"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-8f527027-8fff-4595-952b-45e88c7fc7bc\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>label</th>\n",
              "      <th>message</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>ham</td>\n",
              "      <td>Go until jurong point, crazy.. Available only ...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>ham</td>\n",
              "      <td>Ok lar... Joking wif u oni...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>spam</td>\n",
              "      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>ham</td>\n",
              "      <td>U dun say so early hor... U c already then say...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>ham</td>\n",
              "      <td>Nah I don't think he goes to usf, he lives aro...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>...</th>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5567</th>\n",
              "      <td>spam</td>\n",
              "      <td>This is the 2nd time we have tried 2 contact u...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5568</th>\n",
              "      <td>ham</td>\n",
              "      <td>Will ü b going to esplanade fr home?</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5569</th>\n",
              "      <td>ham</td>\n",
              "      <td>Pity, * was in mood for that. So...any other s...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5570</th>\n",
              "      <td>ham</td>\n",
              "      <td>The guy did some bitching but I acted like i'd...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5571</th>\n",
              "      <td>ham</td>\n",
              "      <td>Rofl. Its true to its name</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>5572 rows × 2 columns</p>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-8f527027-8fff-4595-952b-45e88c7fc7bc')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-8f527027-8fff-4595-952b-45e88c7fc7bc button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-8f527027-8fff-4595-952b-45e88c7fc7bc');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "    <div id=\"df-7fd5817f-cc65-4f6f-ae1d-54cb0b6d46bf\">\n",
              "      <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-7fd5817f-cc65-4f6f-ae1d-54cb0b6d46bf')\"\n",
              "                title=\"Suggest charts\"\n",
              "                style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "      </button>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "      --bg-color: #E8F0FE;\n",
              "      --fill-color: #1967D2;\n",
              "      --hover-bg-color: #E2EBFA;\n",
              "      --hover-fill-color: #174EA6;\n",
              "      --disabled-fill-color: #AAA;\n",
              "      --disabled-bg-color: #DDD;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "      --bg-color: #3B4455;\n",
              "      --fill-color: #D2E3FC;\n",
              "      --hover-bg-color: #434B5C;\n",
              "      --hover-fill-color: #FFFFFF;\n",
              "      --disabled-bg-color: #3B4455;\n",
              "      --disabled-fill-color: #666;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart {\n",
              "    background-color: var(--bg-color);\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: var(--fill-color);\n",
              "    height: 32px;\n",
              "    padding: 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: var(--hover-bg-color);\n",
              "    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: var(--button-hover-fill-color);\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart-complete:disabled,\n",
              "  .colab-df-quickchart-complete:disabled:hover {\n",
              "    background-color: var(--disabled-bg-color);\n",
              "    fill: var(--disabled-fill-color);\n",
              "    box-shadow: none;\n",
              "  }\n",
              "\n",
              "  .colab-df-spinner {\n",
              "    border: 2px solid var(--fill-color);\n",
              "    border-color: transparent;\n",
              "    border-bottom-color: var(--fill-color);\n",
              "    animation:\n",
              "      spin 1s steps(1) infinite;\n",
              "  }\n",
              "\n",
              "  @keyframes spin {\n",
              "    0% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "      border-left-color: var(--fill-color);\n",
              "    }\n",
              "    20% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    30% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    40% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    60% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    80% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "    90% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "  }\n",
              "</style>\n",
              "\n",
              "      <script>\n",
              "        async function quickchart(key) {\n",
              "          const quickchartButtonEl =\n",
              "            document.querySelector('#' + key + ' button');\n",
              "          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.\n",
              "          quickchartButtonEl.classList.add('colab-df-spinner');\n",
              "          try {\n",
              "            const charts = await google.colab.kernel.invokeFunction(\n",
              "                'suggestCharts', [key], {});\n",
              "          } catch (error) {\n",
              "            console.error('Error during call to suggestCharts:', error);\n",
              "          }\n",
              "          quickchartButtonEl.classList.remove('colab-df-spinner');\n",
              "          quickchartButtonEl.classList.add('colab-df-quickchart-complete');\n",
              "        }\n",
              "        (() => {\n",
              "          let quickchartButtonEl =\n",
              "            document.querySelector('#df-7fd5817f-cc65-4f6f-ae1d-54cb0b6d46bf button');\n",
              "          quickchartButtonEl.style.display =\n",
              "            google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "        })();\n",
              "      </script>\n",
              "    </div>\n",
              "\n",
              "  <div id=\"id_29b83a0d-41c2-4bf2-bf7a-7cafd88a4ffd\">\n",
              "    <style>\n",
              "      .colab-df-generate {\n",
              "        background-color: #E8F0FE;\n",
              "        border: none;\n",
              "        border-radius: 50%;\n",
              "        cursor: pointer;\n",
              "        display: none;\n",
              "        fill: #1967D2;\n",
              "        height: 32px;\n",
              "        padding: 0 0 0 0;\n",
              "        width: 32px;\n",
              "      }\n",
              "\n",
              "      .colab-df-generate:hover {\n",
              "        background-color: #E2EBFA;\n",
              "        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "        fill: #174EA6;\n",
              "      }\n",
              "\n",
              "      [theme=dark] .colab-df-generate {\n",
              "        background-color: #3B4455;\n",
              "        fill: #D2E3FC;\n",
              "      }\n",
              "\n",
              "      [theme=dark] .colab-df-generate:hover {\n",
              "        background-color: #434B5C;\n",
              "        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "        fill: #FFFFFF;\n",
              "      }\n",
              "    </style>\n",
              "    <button class=\"colab-df-generate\" onclick=\"generateWithVariable('messages')\"\n",
              "            title=\"Generate code using this dataframe.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "    <script>\n",
              "      (() => {\n",
              "      const buttonEl =\n",
              "        document.querySelector('#id_29b83a0d-41c2-4bf2-bf7a-7cafd88a4ffd button.colab-df-generate');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      buttonEl.onclick = () => {\n",
              "        google.colab.notebook.generateWithVariable('messages');\n",
              "      }\n",
              "      })();\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "dataframe",
              "variable_name": "messages",
              "summary": "{\n  \"name\": \"messages\",\n  \"rows\": 5572,\n  \"fields\": [\n    {\n      \"column\": \"label\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 2,\n        \"samples\": [\n          \"spam\",\n          \"ham\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"message\",\n      \"properties\": {\n        \"dtype\": \"string\",\n        \"num_unique_values\": 5169,\n        \"samples\": [\n          \"K, makes sense, btw carlos is being difficult so you guys are gonna smoke while I go pick up the second batch and get gas\",\n          \"URGENT! Your mobile No *********** WON a \\u00a32,000 Bonus Caller Prize on 02/06/03! This is the 2nd attempt to reach YOU! Call 09066362220 ASAP! BOX97N7QP, 150ppm\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}"
            }
          },
          "metadata": {},
          "execution_count": 7
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import re\n",
        "import nltk\n",
        "nltk.download('stopwords')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "__ee7AS8XtGb",
        "outputId": "908bddc0-d03d-4e29-ffa2-378d43baad1d"
      },
      "execution_count": 8,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "[nltk_data] Downloading package stopwords to /root/nltk_data...\n",
            "[nltk_data]   Package stopwords is already up-to-date!\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "True"
            ]
          },
          "metadata": {},
          "execution_count": 8
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from nltk.corpus import stopwords\n",
        "from nltk.stem.porter import PorterStemmer"
      ],
      "metadata": {
        "id": "3eZxsv14YaEr"
      },
      "execution_count": 3,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ps=PorterStemmer()"
      ],
      "metadata": {
        "id": "f03SGD9VZAis"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "corpus=[]\n",
        "for i in range(len(messages)):\n",
        "  review=re.sub('[^a-zA-Z]',' ',messages['message'][i])\n",
        "  review=review.lower()\n",
        "  review=review.split()\n",
        "  review=[ps.stem(word)for word in review if word not in stopwords.words('english')]\n",
        "  review=''.join(review)\n",
        "  corpus.append(review)"
      ],
      "metadata": {
        "id": "fEc25KS4ZE18"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Bag of Word"
      ],
      "metadata": {
        "id": "3zGOjQLjzf7B"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.feature_extraction.text import CountVectorizer\n",
        "cv=CountVectorizer(max_features=2500,binary=True)\n",
        "x=cv.fit_transform(corpus).toarray()"
      ],
      "metadata": {
        "id": "dg3-rf9qcIBR"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "x[0]"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Y5GkXlC0zQuo",
        "outputId": "70dcaa44-d21d-4931-99f3-1309fb0c8832"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([0, 0, 0, ..., 0, 0, 0])"
            ]
          },
          "metadata": {},
          "execution_count": 13
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "encoding y"
      ],
      "metadata": {
        "id": "UILkXX_Rzikv"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "y=pd.get_dummies(messages['label'])\n",
        "y=y.iloc[:,1].values.astype(int)\n",
        "y"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "C2-OWwpfzR_U",
        "outputId": "cd8aa92a-0e05-4959-c836-87a3d966ce36"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([0, 0, 1, ..., 0, 0, 0])"
            ]
          },
          "metadata": {},
          "execution_count": 19
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.model_selection import train_test_split\n"
      ],
      "metadata": {
        "id": "6f7-xP6SaKKU"
      },
      "execution_count": 4,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#train test split\n",
        "x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)"
      ],
      "metadata": {
        "id": "MVPrCwos0tsy"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Model building"
      ],
      "metadata": {
        "id": "xSprFRcq1ab_"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.naive_bayes import MultinomialNB"
      ],
      "metadata": {
        "id": "7LOYj4Rt1aH-"
      },
      "execution_count": 27,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "model=MultinomialNB().fit(x_train,y_train)"
      ],
      "metadata": {
        "id": "JrpOimmk1FXJ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "pred=model.predict(x_test)"
      ],
      "metadata": {
        "id": "rLD8hToj2AGf"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.metrics import accuracy_score,classification_report"
      ],
      "metadata": {
        "id": "_DmDyZV82EMd"
      },
      "execution_count": 5,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "acc=accuracy_score(pred,y_test)\n",
        "acc"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "pgAUOoWq2OdQ",
        "outputId": "a203a9c0-e4e6-4aeb-8767-9660b3acb680"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.8609865470852018"
            ]
          },
          "metadata": {},
          "execution_count": 29
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "report=classification_report(pred,y_test)\n",
        "print(report)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ICQSMcEk2hb0",
        "outputId": "b6a95d15-46b3-46c7-ffdb-9eff107381cd"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "              precision    recall  f1-score   support\n",
            "\n",
            "           0       1.00      0.86      0.93      1112\n",
            "           1       0.02      1.00      0.04         3\n",
            "\n",
            "    accuracy                           0.86      1115\n",
            "   macro avg       0.51      0.93      0.48      1115\n",
            "weighted avg       1.00      0.86      0.92      1115\n",
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "TF-IDF"
      ],
      "metadata": {
        "id": "2IUtUub42-4S"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.feature_extraction.text import TfidfVectorizer"
      ],
      "metadata": {
        "id": "wzKo7WFM3BAP"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "tf=TfidfVectorizer(max_features=2500)"
      ],
      "metadata": {
        "id": "Vnif-Ex23MtG"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "x=tf.fit_transform(corpus).toarray()"
      ],
      "metadata": {
        "id": "iypyBo7z3X36"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)"
      ],
      "metadata": {
        "id": "aqxQbLJw3ndL"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "model1=MultinomialNB().fit(x_train,y_train)"
      ],
      "metadata": {
        "id": "a_FYNUEN35I6"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "pred=model1.predict(x_test)"
      ],
      "metadata": {
        "id": "tzrpWYyH4JxA"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "acc=accuracy_score(pred,y_test)\n",
        "acc"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "zjof3khT4Xex",
        "outputId": "9dc50286-0bea-4576-b537-a2bbd6974e74"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.8609865470852018"
            ]
          },
          "metadata": {},
          "execution_count": 57
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "report=classification_report(pred,y_test)\n",
        "print(report)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "k4rAH3Nc4bLk",
        "outputId": "e3434844-f23a-4083-8867-8f9f7af15798"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "              precision    recall  f1-score   support\n",
            "\n",
            "           0       1.00      0.86      0.93      1112\n",
            "           1       0.02      1.00      0.04         3\n",
            "\n",
            "    accuracy                           0.86      1115\n",
            "   macro avg       0.51      0.93      0.48      1115\n",
            "weighted avg       1.00      0.86      0.92      1115\n",
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Word2Vec"
      ],
      "metadata": {
        "id": "zdH6jacD3Q_S"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install gensim"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "e8fTfLwZ3Qku",
        "outputId": "a98cbe00-e9b3-4ce5-d1cc-0ed7d2d518e9"
      },
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: gensim in /usr/local/lib/python3.11/dist-packages (4.3.3)\n",
            "Requirement already satisfied: numpy<2.0,>=1.18.5 in /usr/local/lib/python3.11/dist-packages (from gensim) (1.26.4)\n",
            "Requirement already satisfied: scipy<1.14.0,>=1.7.0 in /usr/local/lib/python3.11/dist-packages (from gensim) (1.13.1)\n",
            "Requirement already satisfied: smart-open>=1.8.1 in /usr/local/lib/python3.11/dist-packages (from gensim) (7.3.0.post1)\n",
            "Requirement already satisfied: wrapt in /usr/local/lib/python3.11/dist-packages (from smart-open>=1.8.1->gensim) (1.17.3)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import gensim.downloader as api"
      ],
      "metadata": {
        "id": "t1VjZ4iv8Jz2"
      },
      "execution_count": 1,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "wv=api.load('word2vec-google-news-300')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "22EqOnce8QFA",
        "outputId": "b48a26c4-a980-48a7-d5b7-7018ad07cc60"
      },
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[==================================================] 100.0% 1662.8/1662.8MB downloaded\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "nltk.download('wordnet')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "mAWtBg7S4jMo",
        "outputId": "e3bae269-6159-4042-9ea4-de0ab9600e84"
      },
      "execution_count": 9,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "[nltk_data] Downloading package wordnet to /root/nltk_data...\n",
            "[nltk_data]   Package wordnet is already up-to-date!\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "True"
            ]
          },
          "metadata": {},
          "execution_count": 9
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from nltk.stem import WordNetLemmatizer\n",
        "lemmentize=WordNetLemmatizer()"
      ],
      "metadata": {
        "id": "KODcoxRF3U6N"
      },
      "execution_count": 10,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "corpus=[]\n",
        "for i in range(len(messages)):\n",
        "  review=re.sub('[^a-zA-Z]',' ',messages['message'][i])\n",
        "  review=review.lower()\n",
        "  review=review.split()\n",
        "  review=[lemmentize.lemmatize(word)for word in review if word not in stopwords.words('english')]\n",
        "  review=' '.join(review)\n",
        "  corpus.append(review)"
      ],
      "metadata": {
        "id": "4e5o_dh94BO6"
      },
      "execution_count": 11,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "corpus[0]"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 35
        },
        "id": "DLdKoAIh4U7l",
        "outputId": "be42a306-eb1c-4933-e61c-1b26110a56be"
      },
      "execution_count": 11,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "'go jurong point crazy available bugis n great world la e buffet cine got amore wat'"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "string"
            }
          },
          "metadata": {},
          "execution_count": 11
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from nltk import sent_tokenize\n",
        "from gensim.utils import simple_preprocess"
      ],
      "metadata": {
        "id": "sVWqbj8c44mA"
      },
      "execution_count": 12,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "nltk.download('punkt_tab')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "1Ccu9AB356Vn",
        "outputId": "27e210ff-1fd7-4fdc-b6b4-0c7cf8585c12"
      },
      "execution_count": 13,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "[nltk_data] Downloading package punkt_tab to /root/nltk_data...\n",
            "[nltk_data]   Package punkt_tab is already up-to-date!\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "True"
            ]
          },
          "metadata": {},
          "execution_count": 13
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "word=[]\n",
        "for sent in corpus:\n",
        "  token=sent_tokenize(sent)\n",
        "  for sent in token:\n",
        "    word.append(simple_preprocess(sent))"
      ],
      "metadata": {
        "id": "7nbQwgG15QJ2"
      },
      "execution_count": 14,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "word"
      ],
      "metadata": {
        "collapsed": true,
        "id": "EQ30k-dT53rQ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import gensim"
      ],
      "metadata": {
        "id": "idOhEPyq58ym"
      },
      "execution_count": 15,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "model=gensim.models.Word2Vec(word,window=5,min_count=2)"
      ],
      "metadata": {
        "id": "8iXXdltS7gEV"
      },
      "execution_count": 16,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "model.wv.similar_by_word('prize')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "jOD2JRhE7ptD",
        "outputId": "aed53a15-8f17-4bc4-86f5-75dad24fd177"
      },
      "execution_count": 18,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "[('claim', 0.9993538856506348),\n",
              " ('line', 0.9992361068725586),\n",
              " ('call', 0.9992241263389587),\n",
              " ('cash', 0.9991540908813477),\n",
              " ('draw', 0.9991230368614197),\n",
              " ('show', 0.9990366697311401),\n",
              " ('mobile', 0.9989935159683228),\n",
              " ('land', 0.9989557862281799),\n",
              " ('free', 0.9989538192749023),\n",
              " ('service', 0.99891597032547)]"
            ]
          },
          "metadata": {},
          "execution_count": 18
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy as np"
      ],
      "metadata": {
        "id": "h4aFm-UkXn42"
      },
      "execution_count": 17,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def avg_word2vec(doc):\n",
        "  valid=[model.wv[word] for word in doc if word in model.wv.index_to_key]\n",
        "  if len(valid)>0:\n",
        "    return np.mean(valid,axis=0)\n",
        "  else:\n",
        "    return np.zeros(model.vector_size)\n"
      ],
      "metadata": {
        "id": "H5b96oKwXEXX"
      },
      "execution_count": 18,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install tqdm\n",
        "from tqdm import tqdm"
      ],
      "metadata": {
        "id": "JuU18A8u9Bfg",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "605e8c50-ff5a-44a6-f149-db8b8a6a9450"
      },
      "execution_count": 19,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: tqdm in /usr/local/lib/python3.11/dist-packages (4.67.1)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "x=[]\n",
        "for i in tqdm(range(len(word))):\n",
        "  x.append(avg_word2vec(word[i]))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "fa5c5CO7U5ng",
        "outputId": "a15c92b0-f8bf-4724-fecf-5e6c8abef2c3"
      },
      "execution_count": 20,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "100%|██████████| 5564/5564 [00:02<00:00, 2587.04it/s]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "type(x)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "-RqaP6HDbZHg",
        "outputId": "8b245b60-c37f-478f-c63d-3813aa82b98c"
      },
      "execution_count": 48,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "list"
            ]
          },
          "metadata": {},
          "execution_count": 48
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "x_new=np.array(x)"
      ],
      "metadata": {
        "id": "cL5r9T9haaop"
      },
      "execution_count": 21,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "x_new.shape"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "0peA4WOkXsce",
        "outputId": "c8050ea9-4fae-45d1-e72d-cc94a9476cb5"
      },
      "execution_count": 55,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(5564, 100)"
            ]
          },
          "metadata": {},
          "execution_count": 55
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "y=messages.iloc[:,1]"
      ],
      "metadata": {
        "id": "eD5FRtf8XtHl"
      },
      "execution_count": 22,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "y=y[:len(x_new)]"
      ],
      "metadata": {
        "id": "NZWY8EKWdEEL"
      },
      "execution_count": 23,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "x_train,x_test,y_train,y_test=train_test_split(x_new,y,test_size=0.2,random_state=2)"
      ],
      "metadata": {
        "id": "dTb3jvfdaA8l"
      },
      "execution_count": 35,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.ensemble import RandomForestClassifier"
      ],
      "metadata": {
        "id": "dr7fKFmcdxrF"
      },
      "execution_count": 25,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "fin_model=RandomForestClassifier().fit(x_train,y_train)"
      ],
      "metadata": {
        "id": "u1tHXHAeaWgP"
      },
      "execution_count": 36,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "pred=fin_model.predict(x_test)"
      ],
      "metadata": {
        "id": "Sb86JtsAdc4l"
      },
      "execution_count": 42,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "acc=accuracy_score(pred,y_test)"
      ],
      "metadata": {
        "id": "_J1nOVJjd8d_"
      },
      "execution_count": 43,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "acc"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "wwU0sDKChTL4",
        "outputId": "c6b7b57d-9807-42e8-b631-a72505d02c3c"
      },
      "execution_count": 44,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.0"
            ]
          },
          "metadata": {},
          "execution_count": 44
        }
      ]
    }
  ]
}